"""
Kjører hele pipelinen i riktig rekkefølge, med tester før og etter.

  python run_all.py                  # tester, så alle steg fra features.py til evaluate_extra.py, så tester igjen
  python run_all.py --list           # vis stegene
  python run_all.py --resume         # fortsett der forrige kjøring stoppet (feil, Ctrl+C eller lukket vindu)
  python run_all.py --from realtime  # start fra et bestemt steg
  python run_all.py --only finalize,rt_breakdown
  python run_all.py --reuse          # features.py --reuse: gjenbruk clean-tabellen i data/features.duckdb
  python run_all.py --checks         # også kontrollene av evalueringen (tar flere timer ekstra)
  python run_all.py --no-tests

Hvert steg logges til logs/<tidspunkt>_<steg>.log og vises samtidig i terminalen. Stopper ved første feil og
skriver hvordan du fortsetter. Rådata (retrieval.py) og vær (src/weather.py) hentes ikke på nytt.

Stoppe og fortsette senere: trykk Ctrl+C, eller lag en tom fil logs/STOPP (for terminaler der Ctrl+C ikke kommer
fram, f.eks. kjørevinduet i noen IDE-er). Steget som kjører avsluttes da innen et sekund, sammen med alle
prosessene det har startet. Fortsett med --resume; det avbrutte steget kjøres da på nytt fra starten, men
rolling_eval, ensemble, select_rt og ablation lagrer underveis og hopper over det som er ferdig (så lenge
datasettene ikke er bygget på nytt i mellomtiden).
"""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (navn, argumenter, ekstra miljøvariabler, beskrivelse)
STEPS = [
    # Tallene i README sjekkes bare etterpå: før en ny kjøring er det vanlig at de ikke stemmer ennå
    ("tests_before", ["-m", "pytest", "-q", "--ignore=tests/test_readme_numbers.py"], {},
     "tester (enhet, lekkasje, app)"),
    ("features", ["src/features.py"], {}, "DuckDB: rensing, historikk, train/valid/test"),
    ("train", ["src/train.py"], {}, "modell A + test"),
    ("importance", ["src/importance.py"], {}, "permutasjonsviktighet for modell A (til appen)"),
    ("rolling_eval", ["src/rolling_eval.py"], {}, "rullende evaluering (ferdige måneder hoppes over)"),
    ("ensemble", ["src/ensemble.py"], {}, "modell A: seed-snitt og blanding"),
    ("realtime", ["src/realtime.py"], {}, "sanntidsfeatures -> data/rt_*.parquet"),
    ("select_rt", ["src/select_rt.py"], {}, "modell B: valg av variant på sep-okt"),
    ("train_rt_r1", ["src/train_rt.py", "--r1"], {"SEEDS": "42"}, "modell B runde 1 (én seed, som originalen)"),
    ("train_rt", ["src/train_rt.py"], {}, "modell B: 3 seeds + test per lead"),
    ("train_rt_nort", ["src/train_rt.py", "--no-rt"], {}, "kontroll: B uten sanntidsfeatures"),
    ("finalize", ["src/finalize.py"], {}, "endelige tall (reports/final.json)"),
    ("oracle", ["src/oracle.py"], {}, "orakel-analyse"),
    ("rt_breakdown", ["src/rt_breakdown.py"], {}, "B mot A (og runde 1) per lead og posisjon"),
    ("evaluate_extra", ["src/evaluate_extra.py"], {}, "AUC/Brier, intervaller, verste dager"),
]
CHECKS = [
    ("check_trip_structure", ["src/check_trip_structure.py"], {}, "avslører avkortede turer noe?"),
    ("rolling_refit", ["src/rolling_eval.py", "--refit", "--min-rounds", "100"], {},
     "rullende evaluering som hovedmodellen (egen fil)"),
    ("ablation_cv", ["src/ablation.py", "--cv", "--seeds", "0,1"], {}, "valg over mars-okt, nå også V4u (ukefolder)"),
    ("ensemble_cv", ["src/ensemble.py"], {}, "modell A: blandingen kontrollert over mars-okt (fra ablation_cv)"),
    ("train_uten_turstruktur", ["src/train.py", "--without", "n_stops,route_frac"], {},
     "modell A uten n_stops og route_frac (hvor mye gir turstrukturen?)"),
]
TESTS_AFTER = ("tests_after", ["-m", "pytest", "-q"], {}, "tester igjen (tallene i README mot nye resultater)")
LOG_DIR = ROOT / "logs"
STATE = LOG_DIR / "run_all_state.json"     # stegene som gjenstår, til --resume
STOP_FILE = LOG_DIR / "STOPP"              # alternativ til Ctrl+C


def save_state(remaining, reuse):
    STATE.write_text(json.dumps({"remaining": remaining, "reuse": reuse}), encoding="utf-8")


# ---------- Prosesshåndtering ----------
# Steget kjøres i en egen prosessgruppe, så Ctrl+C bare treffer run_all.py (ellers kan Ctrl+C bli liggende ubehandlet
# i et steg som er midt i et langt DuckDB- eller LightGBM-kall). run_all.py venter med korte tidsavbrudd, så det
# reagerer innen et sekund, og avslutter da hele steget med alle prosessene det har startet (f.eks. testene, som
# starter features.py og realtime.py). På Windows legges steget i et jobbobjekt, så prosessene også avsluttes hvis
# run_all.py selv blir stoppet eller vinduet lukkes.

def _windows_job():
    """Jobbobjekt med KILL_ON_JOB_CLOSE (Windows). None hvis det ikke lar seg lage."""
    try:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000             # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # Extended limits
            k32.CloseHandle(job)
            return None
        return k32, job
    except Exception:  # noqa: BLE001 - jobbobjektet er en ekstra sikring; taskkill brukes ellers
        return None


def _start(cmd, env):
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP   # Ctrl+C går ikke til steget
    else:
        kw["start_new_session"] = True                              # egen prosessgruppe, ikke terminalens
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
    p.job = None
    if os.name == "nt":
        job = _windows_job()
        if job and job[0].AssignProcessToJobObject(job[1], int(p._handle)):
            p.job = job
        elif job:
            job[0].CloseHandle(job[1])
    return p


def _stop(p):
    """Avslutter steget og alle prosessene det har startet."""
    if os.name == "nt":
        if p.job:
            p.job[0].TerminateJobObject(p.job[1], 1)
        if p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                p.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def _close(p):
    if p.job:
        p.job[0].CloseHandle(p.job[1])
        p.job = None


def run(name, args, env, log_dir):
    log = log_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{name}.log"
    e = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **env)
    shown = " ".join(f"{k}={v}" for k, v in env.items()) + (" " if env else "") + "python " + " ".join(args)
    print(f"\n{'=' * 100}\n>>> {name}: {shown}\n    logg: {log.relative_to(ROOT)}\n{'=' * 100}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"$ {shown}\n")
        p = _start([PY, *args], e)

        def pump():                      # utskriften fra steget, i en egen tråd så hovedtråden kan reagere på Ctrl+C
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    code = p.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if STOP_FILE.exists():
                        STOP_FILE.unlink()
                        print("\nFant logs/STOPP - stopper.", flush=True)
                        raise KeyboardInterrupt from None
        except KeyboardInterrupt:
            print(f"\nStopper '{name}' ...", flush=True)
            while True:              # et nytt Ctrl+C under opprydningen skal ikke etterlate prosesser
                try:
                    _stop(p)
                    reader.join(timeout=5)
                    break
                except KeyboardInterrupt:
                    continue
            f.write("\n*** AVBRUTT ***\n")
            _close(p)
            raise
        reader.join()
        _close(p)
    return code, time.time() - t0


def fmt(sec):
    h, rest = divmod(int(sec), 3600)
    return f"{h}t {rest // 60:02d}m {rest % 60:02d}s" if h else f"{rest // 60}m {rest % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Kjør hele pipelinen.")
    ap.add_argument("--list", action="store_true", help="vis stegene og avslutt")
    ap.add_argument("--resume", action="store_true", help="fortsett der forrige kjøring stoppet")
    ap.add_argument("--from", dest="start", help="start fra dette steget")
    ap.add_argument("--only", help="bare disse stegene (kommaseparert)")
    ap.add_argument("--reuse", action="store_true", help="features.py --reuse")
    ap.add_argument("--checks", action="store_true", help="også kontrollene av evalueringen (flere timer)")
    ap.add_argument("--no-tests", action="store_true", help="hopp over testene før og etter")
    a = ap.parse_args()

    every = STEPS + CHECKS + [TESTS_AFTER]
    if a.reuse:
        every = [(n, args + ["--reuse"] if n == "features" else args, env, d) for n, args, env, d in every]
    if a.list:
        for n, args, env, d in every:
            print(f"  {n:22s} {d}")
        return
    steps = [s for s in every if s[0] in [x[0] for x in STEPS]
             and not (a.no_tests and s[0] == "tests_before")]
    if a.checks:
        steps += [s for s in every if s[0] in [x[0] for x in CHECKS]]
    if not a.no_tests:
        steps.append(every[-1])
    names = [s[0] for s in steps]
    if a.only:
        wanted = a.only.split(",")
        unknown = [w for w in wanted if w not in [s[0] for s in every]]
        if unknown:
            sys.exit(f"Ukjent steg: {', '.join(unknown)} (se python run_all.py --list)")
        steps = [s for s in every if s[0] in wanted]
    elif a.start:
        if a.start not in names:
            sys.exit(f"Ukjent steg: {a.start} (se python run_all.py --list)")
        steps = steps[names.index(a.start):]
    elif a.resume:
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"remaining": []}
        if not state["remaining"]:
            sys.exit("Ingenting å fortsette: forrige kjøring ble fullført (eller ingen kjøring er startet).")
        if state.get("reuse") and not a.reuse:
            every = [(n, args + ["--reuse"] if n == "features" else args, env, d) for n, args, env, d in every]
        steps = [s for name in state["remaining"] for s in every if s[0] == name]

    LOG_DIR.mkdir(exist_ok=True)
    if STOP_FILE.exists():
        STOP_FILE.unlink()               # en gammel stoppfil skal ikke stoppe en ny kjøring
    print(f"Python: {PY}\nSteg: {', '.join(s[0] for s in steps)}\n"
          "Stopp når som helst med Ctrl+C (eller lag filen logs/STOPP), og fortsett med python run_all.py --resume",
          flush=True)
    done, t_all = [], time.time()
    reuse = a.reuse or any("--reuse" in s[1] for s in steps)
    for i, (name, args, env, _) in enumerate(steps):
        save_state([s[0] for s in steps[i:]], reuse)
        try:
            code, sec = run(name, args, env, LOG_DIR)
        except KeyboardInterrupt:
            print(f"\nAvbrutt under '{name}'. Fortsett senere med:\n    python run_all.py --resume\n"
                  f"('{name}' kjøres da på nytt fra starten av steget.)", flush=True)
            sys.exit(130)
        done.append((name, code, sec))
        if code != 0:
            if name == "tests_after":
                save_state([], reuse)
                print("\nTestene etter kjøringen feilet. Har resultatene endret seg, viser test_readme_numbers hvilke tall i "
                      "README.md og docs/experiments.md som må oppdateres.")
                break
            print(f"\nSteget '{name}' feilet (kode {code}). Se loggen i logs/. Rett feilen og fortsett med:\n"
                  "    python run_all.py --resume")
            break
    else:
        save_state([], reuse)
    print("\nOppsummering:")
    for name, code, sec in done:
        print(f"  {'OK  ' if code == 0 else 'FEIL'}  {name:22s} {fmt(sec)}")
    print(f"  Totalt {fmt(time.time() - t_all)}")
    if any(code != 0 for name, code, _ in done if name != "tests_after"):
        sys.exit(1)


if __name__ == "__main__":
    main()
