import matplotlib.pyplot as plt

# RESULTS
# -------------------------------------------
# PHC-WOLF:
# Winning possibility: 0.43301999999999996
# Standard Deviation: 0.03754819835890931
# -------------------------------------------
# Ardavan — Yesterday at 11:11 PM
# PHC-WOLF 2X:
# Winning possibility: 0.45670700000000003
# Standard Deviation: 0.0357809877625417
# -------------------------------------------
# MINMAX-Q:
# Winning possibility: 0.375
# Standard Deviation: 0

PHC_WOLFx1_label = "PHC WOLF 1x"
PHC_WOLFx1_winning_probability = 43.301999999999996
PHC_WOLFx1_standard_deviation = 3.754819835890931
PHC_WOLFx1_color = "gray"

PHC_WOLFx2_label = "PHC WOLF 2x"
PHC_WOLFx2_winning_probability = 45.670700000000003
PHC_WOLFx2_standard_deviation = 3.57809877625417
PHC_WOLFx2_color = "gray"

MINMAX_Q_label = "MINMAX Q"
MINMAX_Q_winning_probability = 37.5
MINMAX_Q_standard_deviation = 0.1 # Just to show a black dot on the graph
MINMAX_Q_color = "gray"

probabilities = [PHC_WOLFx1_winning_probability, PHC_WOLFx2_winning_probability, MINMAX_Q_winning_probability]
stds = [PHC_WOLFx1_standard_deviation, PHC_WOLFx2_standard_deviation, MINMAX_Q_standard_deviation]
labels = [PHC_WOLFx1_label, PHC_WOLFx2_label, MINMAX_Q_label]
colors = [PHC_WOLFx1_color, PHC_WOLFx2_color, MINMAX_Q_color]

fig, ax = plt.subplots()
ax.axhline(y=10, linestyle=':', color='gray')
ax.axhline(y=20, linestyle=':', color='gray')
ax.axhline(y=30, linestyle=':', color='gray')
ax.axhline(y=40, linestyle=':', color='gray')
ax.axhline(y=50, linestyle=':', color='gray')

ax.bar(range(len(probabilities)), probabilities, yerr=stds, width=0.4, align='center', color=colors)

ax.set_ylabel('Probability of winning (%)')
ax.set_xticks(range(len(probabilities)))
ax.set_xticklabels(labels)
ax.set_ylim(0, 50)
plt.show()
