# Data

`raw/season-YYYY.csv`: every Premier League match from 1993-94 to 2025-26 (one file per season).

- Source: [Football-Data.co.uk](https://www.football-data.co.uk/), via the
  [datasets/football-datasets](https://github.com/datasets/football-datasets) mirror.
- License: [Open Data Commons Public Domain Dedication and License (PDDL)](http://opendatacommons.org/licenses/pddl/).
- Shots, shots on target and corners are only available from 2000-01 onward, so models train from 2000-01;
  the earlier seasons are used to warm up the Elo ratings.

To refresh with new matches, replace the files with the latest ones from the mirror and rerun `make all`.
