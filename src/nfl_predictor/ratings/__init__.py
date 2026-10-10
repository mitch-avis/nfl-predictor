"""Team and quarterback ratings from opponent-adjusted EPA, ported from ``nfl-sos-ratings``.

The method is the head-to-head-excluded opponent profiling of the separate, read-only
``nfl-sos-ratings`` project and the simultaneous ridge that generalizes it, ported from its
commit ``9081bce`` so that this package reproduces that project's published ratings files:

- :mod:`.ridge`: the weighted offense-versus-defense ridge solver every rating shares.
- :mod:`.inputs`: the team-game and quarterback-game rows every fit consumes.
- :mod:`.team_rating`: points-per-game team ratings (offense, defense, special teams), the
  head-to-head-excluded schedule strength ``sos``, and the week-by-week rating history.

Nothing here reads files or the network: every function takes Polars frames and returns them.
"""
