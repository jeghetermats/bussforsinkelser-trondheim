SELECT
  datedServiceJourneyId AS journey_id,
  operatingDate,
  lineRef,
  directionRef,
  stopPointRef,
  stopPointName,
  CAST(sequenceNr AS INT64) AS seq,
  DATETIME(aimedArrivalTime, "Europe/Oslo") AS aimed_arrival_local,
  TIMESTAMP_DIFF(arrivalTime, aimedArrivalTime, SECOND) AS arrival_delay_s,
  TIMESTAMP_DIFF(departureTime, arrivalTime, SECOND)    AS dwell_s,
  -- forsinkelse ved forrige stopp på samme tur (sterk prediktor)
  LAG(TIMESTAMP_DIFF(arrivalTime, aimedArrivalTime, SECOND))
    OVER (PARTITION BY datedServiceJourneyId ORDER BY CAST(sequenceNr AS INT64)) AS prev_stop_delay_s
FROM `ent-data-sharing-ext-prd.realtime_siri_et.realtime_siri_et_last_recorded`
WHERE operatingDate BETWEEN DATE "{start}" AND DATE "{end}"
  AND dataSource = "ATB"
  AND vehicleMode = "bus"
  AND CAST(journeyCancellation AS STRING) = "false"
  AND CAST(stopCancellation AS STRING)    = "false"
  AND CAST(extraJourney AS STRING)        = "false"
  AND CAST(estimated AS STRING)           = "false"   -- kun faktisk registrerte tider
  AND arrivalTime IS NOT NULL
  AND aimedArrivalTime IS NOT NULL