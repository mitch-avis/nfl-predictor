# FiveThirtyEight Elo fixture

`nfl_games_2017_2019.csv` is every row of seasons 2017, 2018 and 2019 (801 games, playoffs
included), with the header, cut unchanged from `data/nfl_games.csv` in FiveThirtyEight's
`fivethirtyeight/nfl-elo-game` repository at commit `fbec1af` (file SHA-256
`af16cf3c5848393057fad07feea278f0a85ab7b57ddbfe42ed96b5c80e7d519d`). That repository is
published under the MIT License, `Copyright (c) 2021 ABC News Internet Ventures`; the full
notice is in [`LICENSE`](LICENSE) beside this file, as the license requires for a copy.

The cut was made with:

```bash
head -1 nfl-elo-game/data/nfl_games.csv > nfl_games_2017_2019.csv
awk -F, 'NR>1 && ($2=="2017" || $2=="2018" || $2=="2019")' \
    nfl-elo-game/data/nfl_games.csv >> nfl_games_2017_2019.csv
```

Columns: `team1` is the home team (the nominal home team at a neutral site), `elo1` and `elo2`
are the two teams' published pre-game ratings, `elo_prob1` is FiveThirtyEight's pre-game
probability that `team1` wins, and `result1` is `1`, `0.5` or `0` for a `team1` win, tie or
loss. The same numbers appear as `elo1_pre`, `elo2_pre` and `elo_prob1` in FiveThirtyEight's
`nfl_elo.csv`.

`tests/test_elo_ratings.py` seeds every team with its first published rating in the file and
checks that this repository's Elo, with only FiveThirtyEight's public framework switched on,
reproduces every published pre-game rating and probability to floating-point precision. These
seasons were chosen because FiveThirtyEight published full-precision ratings from 2015 on (three
decimals before), and because they hold two offseason reversions, three ties, London and Mexico
City games and three Super Bowls.
