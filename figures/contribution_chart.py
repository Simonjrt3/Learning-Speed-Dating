import matplotlib.pyplot as plt

labels = ["Demographic", "Perceptions", "Preferences"]
influences = [19.59, 57.59, 22.82]

plt.figure(figsize=(7, 7))

plt.pie(
    influences,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False
)

plt.title("Relative Influence on Match")

plt.tight_layout()

# Save high-resolution versions
plt.savefig("relative_influence.png", dpi=300, bbox_inches="tight")
plt.savefig("relative_influence.pdf", bbox_inches="tight")

plt.show()