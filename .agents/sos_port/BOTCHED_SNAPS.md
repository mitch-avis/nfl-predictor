# The botched-snap charge rule in the quarterback rating: measurement

Planned in `protocols/botched_snaps.md` before measuring. The rule: a botched snap counts against a
quarterback only when the official scorer charged the fumble to him. The approved change removes the
aborted snaps of 2006-2025 the scorer charged to a lineman that went on as a pass or sack; the
ported rating counts them today as the passer's dropbacks. Nothing is adopted here.

## Headline

- The approved change removes 60 plays of 2006-2025 (-21.3 EPA by `epa`, -21.3 by `qb_epa`, the
  rating's EPA), against the source's count of 60 at -21.3.
- Full-season ratings (each season's own penalty, as published): 1090 passer-seasons move by more
  than 1e-09, by at most 0.0080 EPA per dropback; 18 ranks among qualified passers change, by at
  most 1.
- The published weekly history moves by at most 0.0112.
- Pre-week ratings (the model's `ratings_qb` and `ratings_qb_dropbacks`), weeks of 2006-2026: 11,742
  of 19,881 passer-weeks move beyond 1e-12, by at most 0.0112; 504 passer-weeks change their
  dropback count. A season's removals also move the next season's pre-week ratings through its
  penalty.

## By season

Movement is measured with the approved change (2006-2025 plays only); 1999-2005 and 2026 are counted
but not removed. A dash marks a part the season's change does not reach (no play removed that
season, or no pre-week week measured).

| season | plays | epa | qb_epa | passers moved | largest change | penalty before | penalty after | rank changes | history largest | pre-week moved | pre-week rows | pre-week largest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1999 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | - | - | - |
| 2000 | 3 | +0.66 | +0.66 | - | - | - | - | - | - | - | - | - |
| 2001 | 1 | -1.18 | -1.18 | - | - | - | - | - | - | - | - | - |
| 2002 | 5 | -3.60 | -3.60 | - | - | - | - | - | - | - | - | - |
| 2003 | 2 | -2.09 | -2.09 | - | - | - | - | - | - | - | - | - |
| 2004 | 1 | -0.58 | -0.58 | - | - | - | - | - | - | - | - | - |
| 2005 | 2 | -0.36 | -0.36 | - | - | - | - | - | - | - | - | - |
| 2006 | 2 | +2.67 | +2.67 | 70 | 0.0030 | 316.228 | 316.228 | 2 | 0.0079 | 890 | 890 | 0.0079 |
| 2007 | 4 | +1.03 | +1.03 | 82 | 0.0054 | 177.828 | 177.828 | 0 | 0.0059 | 591 | 1076 | 0.0038 |
| 2008 | 2 | +0.95 | +0.95 | 74 | 0.0012 | 177.828 | 177.828 | 0 | 0.0018 | 615 | 1006 | 0.0018 |
| 2009 | 1 | +1.78 | +1.78 | 76 | 0.0021 | 100.000 | 100.000 | 0 | 0.0041 | 674 | 1018 | 0.0034 |
| 2010 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | 0 | 996 | 0.0000 |
| 2011 | 3 | -1.81 | -1.81 | 73 | 0.0029 | 177.828 | 177.828 | 0 | 0.0074 | 913 | 948 | 0.0074 |
| 2012 | 3 | -4.52 | -4.52 | 69 | 0.0022 | 316.228 | 316.228 | 2 | 0.0025 | 577 | 868 | 0.0031 |
| 2013 | 4 | -4.01 | -4.01 | 63 | 0.0049 | 177.828 | 177.828 | 2 | 0.0056 | 818 | 886 | 0.0042 |
| 2014 | 5 | +1.03 | +1.03 | 72 | 0.0010 | 316.228 | 316.228 | 0 | 0.0025 | 951 | 985 | 0.0034 |
| 2015 | 10 | +1.59 | +1.59 | 73 | 0.0080 | 316.228 | 316.228 | 2 | 0.0106 | 895 | 895 | 0.0106 |
| 2016 | 1 | -2.01 | -2.01 | 69 | 0.0026 | 177.828 | 177.828 | 4 | 0.0036 | 378 | 900 | 0.0029 |
| 2017 | 7 | -3.09 | -3.09 | 71 | 0.0048 | 316.228 | 316.228 | 2 | 0.0048 | 828 | 899 | 0.0078 |
| 2018 | 6 | -4.59 | -4.59 | 70 | 0.0046 | 177.828 | 177.828 | 0 | 0.0063 | 876 | 913 | 0.0052 |
| 2019 | 8 | -4.18 | -4.18 | 70 | 0.0060 | 177.828 | 177.828 | 2 | 0.0112 | 976 | 976 | 0.0112 |
| 2020 | 2 | -3.59 | -3.59 | 81 | 0.0041 | 316.228 | 316.228 | 0 | 0.0058 | 696 | 1016 | 0.0077 |
| 2021 | 2 | -2.57 | -2.57 | 77 | 0.0022 | 316.228 | 316.228 | 2 | 0.0050 | 1064 | 1064 | 0.0050 |
| 2022 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | 0 | 1069 | 0.0000 |
| 2023 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | 0 | 1092 | 0.0000 |
| 2024 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | 0 | 1061 | 0.0000 |
| 2025 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | 0 | 1101 | 0.0000 |
| 2026 | 0 | +0.00 | +0.00 | - | - | - | - | - | - | 0 | 222 | 0.0000 |

## The plays the approved change removes

| season | game | play | passer | charged | epa | qb_epa | text |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2006 | 2006_01_BUF_NE | 426 | 00-0022912 | 00-0021201 | +2.273 | +2.273 | (9:55) (Shotgun) 7-J.Losman Aborted. 67-M.Fowler FUMBLES at BUF 5, recovered by BUF-7-J.Losman at BUF 5. 7-J.L |
| 2006 | 2006_16_PHI_DAL | 2419 | 00-0021678 | 00-0021385 | +0.402 | +0.402 | (5:32) (Shotgun) 9-T.Romo Aborted. 65-A.Gurode FUMBLES at DAL 19, recovered by DAL-9-T.Romo at DAL 3. 9-T.Romo |
| 2007 | 2007_10_SF_SEA | 161 | 00-0007091 | 00-0023461 | +1.202 | +1.202 | (13:50) (Shotgun) 8-M.Hasselbeck Aborted. 65-C.Spencer FUMBLES at SEA 12, recovered by SEA-8-M.Hasselbeck at S |
| 2007 | 2007_11_WAS_DAL | 564 | 00-0021678 | 00-0021385 | -2.899 | -2.899 | (4:17) (No Huddle, Shotgun) 9-T.Romo Aborted. 65-A.Gurode FUMBLES at DAL 45, recovered by DAL-9-T.Romo at DAL |
| 2007 | 2007_11_WAS_DAL | 1284 | 00-0021678 | 00-0021385 | +1.227 | +1.227 | (6:10) (Shotgun) 9-T.Romo Aborted. 65-A.Gurode FUMBLES at DAL 45, recovered by DAL-9-T.Romo at DAL 45. 9-T.Rom |
| 2007 | 2007_15_CIN_SF | 329 | 00-0020679 | 00-0021341 | +1.502 | +1.502 | (10:25) (Shotgun) 13-S.Hill Aborted. 66-E.Heitmann FUMBLES at SF 21, recovered by SF-13-S.Hill at SF 15. 13-S. |
| 2008 | 2008_09_ARI_STL | 2816 | 00-0017200 | 00-0025682 | +1.170 | +1.170 | (8:17) (Shotgun) 13-K.Warner Aborted. 63-L.Sendlein FUMBLES at ARI 11, recovered by ARI-13-K.Warner at ARI 11. |
| 2008 | 2008_17_KC_CIN | 1884 | 00-0023682 | 00-0023553 | -0.217 | -0.217 | (:09) (Shotgun) 11-R.Fitzpatrick Aborted. 53-E.Ghiaciuc FUMBLES at KC 44, recovered by CIN-11-R.Fitzpatrick at |
| 2009 | 2009_08_MIN_GB | 3413 | 00-0023459 | 00-0022832 | +1.776 | +1.776 | (7:33) (Shotgun) 12-A.Rodgers Aborted. 63-S.Wells FUMBLES at GB 27, recovered by GB-12-A.Rodgers at GB 27. 12- |
| 2011 | 2011_02_DAL_SF | 3001 | 00-0021678 | 00-0027417 | -0.431 | -0.431 | (15:00) (Shotgun) 9-T.Romo Aborted. 67-P.Costa FUMBLES at DAL 7, recovered by DAL-9-T.Romo at DAL 4. 9-T.Romo |
| 2011 | 2011_03_WAS_DAL | 2431 | 00-0021678 | 00-0027417 | -0.886 | -0.886 | (10:31) (Shotgun) 9-T.Romo Aborted. 67-P.Costa FUMBLES at DAL 43, recovered by DAL-9-T.Romo at DAL 25. 9-T.Rom |
| 2011 | 2011_03_WAS_DAL | 3118 | 00-0021678 | 00-0027417 | -0.496 | -0.496 | (14:18) (Shotgun) 9-T.Romo Aborted. 67-P.Costa FUMBLES at WAS 48, recovered by DAL-9-T.Romo at DAL 46. 9-T.Rom |
| 2012 | 2012_08_NYG_DAL | 807 | 00-0021678 | 00-0024266 | -1.590 | -1.590 | (2:33) (Shotgun) 9-T.Romo Aborted. 63-R.Cook FUMBLES at DAL 46, recovered by DAL-9-T.Romo at DAL 33. 9-T.Romo |
| 2012 | 2012_13_TB_DEN | 819 | 00-0010346 | 00-0022019 | -1.341 | -1.341 | (3:01) (Shotgun) 18-P.Manning Aborted. 67-D.Koppen FUMBLES at DEN 20, recovered by DEN-18-P.Manning at DEN 20. |
| 2012 | 2012_14_SD_PIT | 886 | 00-0022924 | 00-0027870 | -1.585 | -1.585 | (15:00) (Shotgun) 7-B.Roethlisberger Aborted. 53-M.Pouncey FUMBLES at PIT 44, recovered by PIT-7-B.Roethlisber |
| 2013 | 2013_03_KC_PHI | 2239 | 00-0020245 | 00-0028129 | -1.667 | -1.667 | (14:56) (No Huddle, Shotgun) 7-M.Vick Aborted. 62-J.Kelce FUMBLES at PHI 19, recovered by PHI-7-M.Vick at PHI |
| 2013 | 2013_03_OAK_DEN | 2622 | 00-0028825 | 00-0027986 | -0.978 | -0.978 | (6:51) (Shotgun) 2-T.Pryor Aborted. 61-S.Wisniewski FUMBLES at LV 0, recovered by LV-2-T.Pryor at LV 0. 2-T.Pr |
| 2013 | 2013_09_PHI_OAK | 4632 | 00-0030419 | 00-0027986 | -0.790 | -0.790 | (1:01) (Shotgun) 14-M.McGloin Aborted. 61-S.Wisniewski FUMBLES at PHI 2, recovered by LV-14-M.McGloin at PHI 5 |
| 2013 | 2013_11_BAL_CHI | 4349 | 00-0026158 | 00-0029268 | -0.578 | -0.578 | (:11) (Shotgun) 5-J.Flacco Aborted. 66-G.Gradkowski FUMBLES at CHI 8, recovered by BAL-5-J.Flacco at CHI 8. 5- |
| 2014 | 2014_02_NYJ_GB | 3031 | 00-0023459 | 00-0031019 | +1.027 | +1.027 | (6:23) (Shotgun) 12-A.Rodgers Aborted. 63-C.Linsley FUMBLES at NYJ 15, recovered by GB-12-A.Rodgers at NYJ 15. |
| 2014 | 2014_06_SD_OAK | 259 | 00-0022942 | 00-0026909 | +0.765 | +0.765 | (12:20) (Shotgun) 17-P.Rivers Aborted. 74-R.Ohrnberger FUMBLES at 50, recovered by LAC-17-P.Rivers at LAC 45. |
| 2014 | 2014_10_SF_NO | 1985 | 00-0020531 | 00-0021267 | +0.410 | +0.410 | (:44) (No Huddle, Shotgun) 9-D.Brees Aborted. 55-J.Goodwin FUMBLES at SF 40, recovered by NO-9-D.Brees at SF 4 |
| 2014 | 2014_14_NYJ_MIN | 267 | 00-0030565 | 00-0024244 | -0.662 | -0.662 | (12:27) (Shotgun) 7-G.Smith Aborted. 74-N.Mangold FUMBLES at MIN 37, recovered by NYJ-7-G.Smith at MIN 37. 7-G |
| 2014 | 2014_14_TB_DET | 2727 | 00-0026498 | 00-0020445 | -0.513 | -0.513 | (2:47) (Shotgun) 9-M.Stafford Aborted. 51-D.Raiola FUMBLES at TB 38, recovered by DET-9-M.Stafford at TB 46. 9 |
| 2015 | 2015_01_DET_SD | 1024 | 00-0022942 | 00-0030991 | -0.905 | -0.905 | (11:38) (Shotgun) 17-P.Rivers Aborted. 65-C.Watt FUMBLES at LAC 14, recovered by LAC-17-P.Rivers at LAC 5. 17- |
| 2015 | 2015_01_NYG_DAL | 4174 | 00-0021678 | 00-0030546 | +3.691 | +3.691 | (:13) (Shotgun) 9-T.Romo Aborted. 72-T.Frederick FUMBLES at NYG 16, recovered by DAL-9-T.Romo at NYG 16. 9-T.R |
| 2015 | 2015_04_KC_CIN | 124 | 00-0027973 | 00-0031306 | +2.583 | +2.583 | (13:19) (Shotgun) 14-A.Dalton Aborted. 61-R.Bodine FUMBLES at CIN 44, recovered by CIN-14-A.Dalton at CIN 44. |
| 2015 | 2015_07_ATL_TEN | 2103 | 00-0026143 | 00-0028177 | -0.667 | -0.667 | (14:00) (Shotgun) 2-M.Ryan Aborted. 68-M.Person FUMBLES at ATL 8, recovered by ATL-2-M.Ryan at ATL 5. 2-M.Ryan |
| 2015 | 2015_09_ATL_SF | 3782 | 00-0026143 | 00-0028177 | +0.550 | +0.550 | (6:15) (Shotgun) 2-M.Ryan Aborted. 68-M.Person FUMBLES at SF 40, recovered by ATL-2-M.Ryan at SF 45. 2-M.Ryan |
| 2015 | 2015_09_TEN_NO | 429 | 00-0032268 | 00-0032118 | -0.599 | -0.599 | (9:04) (Shotgun) 8-M.Mariota Aborted. 69-A.Gallik FUMBLES at TEN 6, recovered by TEN-8-M.Mariota at TEN 6. 8-M |
| 2015 | 2015_11_DAL_MIA | 2494 | 00-0029701 | 00-0027953 | -0.654 | -0.654 | (5:35) (Shotgun) 17-R.Tannehill Aborted. 51-M.Pouncey FUMBLES at DAL 29, recovered by MIA-17-R.Tannehill at DA |
| 2015 | 2015_12_MIA_NYJ | 1586 | 00-0029701 | 00-0031575 | -1.348 | -1.348 | (7:54) (Shotgun) 17-R.Tannehill Aborted. 75-J.Douglas FUMBLES at MIA 36, recovered by MIA-17-R.Tannehill at MI |
| 2015 | 2015_12_MIA_NYJ | 3561 | 00-0029701 | 00-0031575 | +0.280 | +0.280 | (12:00) (Shotgun) 17-R.Tannehill Aborted. 75-J.Douglas FUMBLES at MIA 11, recovered by MIA-17-R.Tannehill at M |
| 2015 | 2015_12_MIA_NYJ | 4567 | 00-0029701 | 00-0031575 | -1.342 | -1.342 | (1:19) (Shotgun) 17-R.Tannehill Aborted. 75-J.Douglas FUMBLES at NYJ 43, recovered by MIA-17-R.Tannehill at NY |
| 2016 | 2016_12_GB_PHI | 3030 | 00-0032950 | 00-0028129 | -2.011 | -2.011 | (11:18) (No Huddle, Shotgun) 11-C.Wentz Aborted. 62-J.Kelce FUMBLES at GB 42, recovered by PHI-11-C.Wentz at G |
| 2017 | 2017_03_KC_LAC | 2876 | 00-0023436 | 00-0030997 | -0.692 | -0.692 | (12:41) (Shotgun) 11-A.Smith Aborted. 73-Z.Fulton FUMBLES at KC 21, recovered by KC-11-A.Smith at KC 21. 11-A. |
| 2017 | 2017_03_PIT_CHI | 4012 | 00-0030520 | 00-0032768 | -0.364 | -0.364 | (1:19) (Shotgun) 8-M.Glennon Aborted. 65-C.Whitehair FUMBLES at CHI 35, recovered by CHI-8-M.Glennon at CHI 27 |
| 2017 | 2017_03_SEA_TEN | 1868 | 00-0029263 | 00-0031282 | -1.579 | -1.579 | (4:24) (Shotgun) 3-R.Wilson Aborted. 68-J.Britt FUMBLES at SEA 19, recovered by SEA-3-R.Wilson at SEA 19. 3-R. |
| 2017 | 2017_04_CIN_CLE | 703 | 00-0027973 | 00-0031306 | +1.267 | +1.267 | (3:55) (Shotgun) 14-A.Dalton Aborted. 61-R.Bodine FUMBLES at CLE 35, recovered by CIN-14-A.Dalton at CLE 35. 1 |
| 2017 | 2017_06_CHI_BAL | 2715 | 00-0033869 | 00-0032768 | -0.731 | -0.731 | (5:02) (Shotgun) 10-M.Trubisky Aborted. 65-C.Whitehair FUMBLES at CHI 1, recovered by CHI-10-M.Trubisky at CHI |
| 2017 | 2017_12_LAC_DAL | 3087 | 00-0033077 | 00-0030546 | +0.538 | +0.538 | (8:56) (No Huddle, Shotgun) 4-D.Prescott Aborted. 72-T.Frederick FUMBLES at LAC 35, recovered by DAL-4-D.Presc |
| 2017 | 2017_17_CLE_PIT | 3463 | 00-0030524 | 00-0029963 | -1.533 | -1.533 | (5:23) (Shotgun) 3-L.Jones Aborted. 74-C.Hubbard FUMBLES at 50, recovered by PIT-3-L.Jones at 50. 3-L.Jones pa |
| 2018 | 2018_02_NYG_DAL | 628 | 00-0033077 | 00-0029278 | +0.075 | +0.075 | (3:13) (Shotgun) 4-D.Prescott Aborted. 73-J.Looney FUMBLES at DAL 15, recovered by DAL-4-D.Prescott at DAL 15. |
| 2018 | 2018_04_SEA_ARI | 3506 | 00-0034343 | 00-0034785 | -0.475 | -0.475 | (10:51) (Shotgun) 3-J.Rosen Aborted. 64-M.Cole FUMBLES at 50, recovered by ARI-3-J.Rosen at ARI 46. 3-J.Rosen |
| 2018 | 2018_05_ARI_SF | 2160 | 00-0034343 | 00-0034785 | -0.822 | -0.822 | (:18) (Shotgun) 3-J.Rosen Aborted. 64-M.Cole FUMBLES at ARI 44, recovered by ARI-3-J.Rosen at ARI 44. 3-J.Rose |
| 2018 | 2018_08_SF_ARI | 4298 | 00-0033936 | 00-0033317 | -1.115 | -1.115 | (:07) (Shotgun) 3-C.Beathard Aborted. 62-E.Magnuson FUMBLES at 50, recovered by SF-3-C.Beathard at SF 34. 3-C. |
| 2018 | 2018_13_ARI_GB | 2370 | 00-0034343 | 00-0034785 | -1.626 | -1.626 | (10:39) (Shotgun) 3-J.Rosen Aborted. 64-M.Cole FUMBLES at ARI 24, recovered by ARI-3-J.Rosen at ARI 24. 3-J.Ro |
| 2018 | 2018_13_ARI_GB | 3758 | 00-0034343 | 00-0034785 | -0.628 | -0.628 | (4:44) (Shotgun) 3-J.Rosen Aborted. 64-M.Cole FUMBLES at ARI 12, recovered by ARI-3-J.Rosen at ARI 12. 3-J.Ros |
| 2019 | 2019_01_PIT_NE | 3179 | 00-0022924 | 00-0027870 | -1.576 | -1.576 | (13:13) (Shotgun) 7-B.Roethlisberger Aborted. 53-M.Pouncey FUMBLES at PIT 49, recovered by PIT-7-B.Roethlisber |
| 2019 | 2019_03_HOU_LAC | 1201 | 00-0033537 | 00-0032969 | +3.034 | +3.034 | (8:18) (Shotgun) 4-D.Watson Aborted. 66-N.Martin FUMBLES at LAC 21, recovered by HOU-4-D.Watson at LAC 21. 4-D |
| 2019 | 2019_05_NYJ_PHI | 344 | 00-0032950 | 00-0028129 | -0.512 | -0.512 | (8:37) (Shotgun) 11-C.Wentz Aborted. 62-J.Kelce FUMBLES at NYJ 28, recovered by PHI-11-C.Wentz at NYJ 28. 11-C |
| 2019 | 2019_06_PIT_LAC | 1948 | 00-0035577 | 00-0027870 | -2.014 | -2.014 | (8:22) (Shotgun) 6-D.Hodges Aborted. 53-M.Pouncey FUMBLES at LAC 8, recovered by PIT-6-D.Hodges at LAC 20. 6-D |
| 2019 | 2019_09_DET_OAK | 396 | 00-0031280 | 00-0035127 | +1.396 | +1.396 | (8:32) (Shotgun) 4-D.Carr Aborted. 68-A.James FUMBLES at DET 27, recovered by LV-4-D.Carr at DET 27. 4-D.Carr |
| 2019 | 2019_09_DET_OAK | 1053 | 00-0031280 | 00-0035127 | -1.360 | -1.360 | (13:42) (Shotgun) 4-D.Carr Aborted. 68-A.James FUMBLES at DET 19, recovered by LV-4-D.Carr at DET 30. 4-D.Carr |
| 2019 | 2019_09_DET_OAK | 3486 | 00-0031280 | 00-0035127 | -1.525 | -1.525 | (7:56) (Shotgun) 4-D.Carr Aborted. 68-A.James FUMBLES at LV 34, recovered by LV-4-D.Carr at LV 29. 4-D.Carr pa |
| 2019 | 2019_13_NYJ_CIN | 3321 | 00-0027973 | 00-0030781 | -1.624 | -1.624 | (11:25) (Shotgun) 14-A.Dalton Aborted. 66-T.Hopkins FUMBLES at CIN 41, recovered by CIN-14-A.Dalton at CIN 38. |
| 2020 | 2020_08_TEN_CIN | 630 | 00-0036442 | 00-0034762 | -0.519 | -0.519 | (4:20) (Shotgun) 9-J.Burrow Aborted. 53-B.Price FUMBLES at CIN 21, recovered by CIN-9-J.Burrow at CIN 21. 9-J. |
| 2020 | 2020_10_HOU_CLE | 348 | 00-0034855 | 00-0030076 | -3.070 | -3.070 | (8:36) 6-B.Mayfield Aborted. 64-J.Tretter FUMBLES at HOU 7, recovered by CLE-6-B.Mayfield at HOU 21. 6-B.Mayfi |
| 2021 | 2021_01_PHI_ATL | 3623 | 00-0036389 | 00-0028129 | -0.518 | -0.518 | (12:53) (No Huddle, Shotgun) 1-J.Hurts Aborted. 62-J.Kelce FUMBLES at PHI 28, recovered by PHI-1-J.Hurts at PH |
| 2021 | 2021_03_MIA_LV | 839 | 00-0031280 | 00-0035127 | -2.057 | -2.057 | (1:36) (Shotgun) 4-D.Carr Aborted. 68-A.James FUMBLES at 50, recovered by LV-4-D.Carr at LV 48. 4-D.Carr pass |

## Outside the approved change (descriptive)

- The same plays outside 2006-2025, which the rule as written also reaches: 2000 3; 2001 1; 2002 5;
  2003 2; 2004 1; 2005 2.
- Aborted snaps of 2006-2025 that went on as a pass or sack, charged to another player (6) or to no
  one (8): the rule's wording would remove these too; the approved change names only the lineman's.
- Attempts stay as they are (they come from the official statistics and decide only the rankings
  qualifier).

## Provenance

- nfl-predictor 0.57.1 at commit `0fdf79cee44dbaec2554315706071ea1bd88d855`; code SHA-256:
  `botched_snaps.py` 4d5b5eaf1c28, `rating_tests_common.py` 50da9bf186e7.
- Inputs: 125 files, fingerprint `781eaf85ab665ffc8e69c93d20e802c2dfd64681fc29285773f0ab8880ef3a8a`
  (each file's SHA-256 is in `botched_snaps.json`), read only, downloads blocked.
- Command: `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/botched_snaps.py
  --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy`; 159 seconds.
