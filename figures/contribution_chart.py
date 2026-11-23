import matplotlib.pyplot as plt

labels = ["Demographic", "Perceptions", "Preferences"]
influences = [19.59, 57.59, 22.82]

plt.figure(figsize=(5, 5))

plt.pie(
    influences,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    textprops={"fontsize": 16}
)

plt.title("Relative Influence on Match", fontsize=20)

plt.tight_layout()

# Save high-resolution versions
plt.savefig("relative_influence.png", dpi=300, bbox_inches="tight")

plt.show()