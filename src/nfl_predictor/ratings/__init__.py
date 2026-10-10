"""Team and quarterback ratings from opponent-adjusted EPA, ported from ``nfl-sos-ratings``.

The method is the head-to-head-excluded opponent profiling of the separate, read-only
``nfl-sos-ratings`` project and the simultaneous ridge that generalizes it, ported from its
commit ``9081bce`` so that this package reproduces that project's published ratings files:

- :mod:`.ridge`: the weighted offense-versus-defense ridge solver every rating shares.

Nothing here reads files or the network: every function takes Polars frames and returns them.
"""
