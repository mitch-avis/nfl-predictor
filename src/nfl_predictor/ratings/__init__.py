"""Team and quarterback ratings from opponent-adjusted EPA, ported from ``nfl-sos-ratings``.

The method is the head-to-head-excluded opponent profiling of the separate, read-only
``nfl-sos-ratings`` project and the simultaneous ridge that generalizes it, ported from its
commit ``9081bce`` so that this package reproduces that project's published ratings files:

- :mod:`.ridge`: the weighted offense-versus-defense ridge solver every rating shares.
- :mod:`.inputs`: the team-game and quarterback-game rows every fit consumes.
- :mod:`.team_rating`: points-per-game team ratings (offense, defense, special teams), the
  head-to-head-excluded schedule strength ``sos``, and the week-by-week rating history.
- :mod:`.team_prior`: the preseason prior inside the team fit, faded out over nine games.
- :mod:`.qb_rating`: the quarterback rating (adjusted EPA per dropback), the
  head-to-head-excluded faced pass defense, and each passer's season volume and ranking
  qualifier.

Nothing here reads files or the network: every function takes Polars frames and returns them.
"""
