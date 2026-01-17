# New TODO Ideas

- Investigate data_collection run:
  - Temporarily change logging level to debug and add helpful debug logging statements throughout
    the relevant data collection code.
  - Determine what takes the entire run so long. Maybe add timers to critical steps to determine
    where potential bottlenecks might be.
  - Is caching already implemented for TeamRankings data and for nflreadypy data? If not, we should
    definitely do so. I believe I already have all the historical TR data saved in `data/`, but I'm
    not sure if that's currently being used or not. I don't see any nflreadpy data saved anywhere
    except for the final outputs of the data collection.
