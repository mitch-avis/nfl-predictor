"""Season statistics tables ported from the separate, read-only ``nfl-sos-ratings`` project.

The tables its web pages show beside the ratings, ported from its commit ``9081bce`` so that this
package reproduces the files that project published: team and quarterback game logs, season rows
(counts per game, rates pooled over the games), head-to-head-excluded opponent profiles, plays and
EPA by win-probability bin, and the metric registry that names and describes every column.

Only :mod:`.sources` reads anything, and only through this repository's loaders; every other
function takes Polars frames and returns them.
"""
