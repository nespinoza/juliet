To test the ligh-travel-time delay (test_transit_and_eclipse_lttd.py), you need to:

- Run gen_transit_eclipse_lttd.py.
- Then run the script with LTTD activated.

Tests constrain the effect to be correct within 40 ppm. Further improvements:

- Implementation calculates time delay from radial distance to the transit location. This should account for the angle of the orbit in reality.
 
