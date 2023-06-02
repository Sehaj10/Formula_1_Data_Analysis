from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting
import pandas as pd

schedule = fastf1.get_event_schedule(2023)
print (schedule)