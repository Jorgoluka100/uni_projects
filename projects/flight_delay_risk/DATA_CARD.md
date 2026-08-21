# Data card — US flight on-time performance

## Source

The project uses monthly **On-Time Reporting Carrier On-Time Performance** files published by the U.S. Department of Transportation Bureau of Transportation Statistics (BTS). The retained portfolio evidence uses January–May 2026.

## Prediction population

The classifier is trained on completed, non-diverted flights with a recorded arrival delay. Cancelled and diverted flights are intentionally excluded because they represent different operational outcomes and should be modelled separately.

## Target

`delay15 = 1` when `ArrDelayMinutes >= 15`, otherwise `0`.

## Information available to the model

Only schedule-time information is used: calendar fields, reporting carrier, origin, destination, route, scheduled departure/arrival time, scheduled elapsed time and distance, plus derived cyclical/time-block features.

Actual departure/arrival times, taxi times, wheels-off/on fields and delay-cause fields are excluded because they would be unavailable at schedule time or leak the outcome.

## Split policy

- Train: January–March 2026
- Validation and alert-threshold selection: April 2026
- Final test: May 2026

The split is temporal rather than random so that evaluation better reflects scoring future flights.

## Limitations

The data does not include the full operational context needed for a production airline system. Weather, aircraft rotation, crew constraints, airport congestion and live network conditions are absent. Results therefore measure the value of schedule-time information, not the ceiling of flight-delay prediction.
