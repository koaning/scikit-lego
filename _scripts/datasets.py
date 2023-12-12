# --8<-- [start:imports]
from collections import Counter

import matplotlib.pylab as plt
import seaborn as sns

sns.set_theme()
# --8<-- [end:imports]


from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

######################################## Abalone #########################################
##########################################################################################

# --8<-- [start:load-abalone]
from sklego.datasets import load_abalone

df_abalone = load_abalone(as_frame=True)
df_abalone.head()
# --8<-- [end:load-abalone]

with open(_static_path / "abalone.md", "w") as f:
    f.write(df_abalone.head().to_markdown(index=False))

# --8<-- [start:plot-abalone]
X, y = load_abalone(return_X_y=True)

plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Distribution of sex (target)")
# --8<-- [end:plot-abalone]

plt.savefig(_static_path / "abalone.png")
plt.clf()


######################################## Arrests #########################################
##########################################################################################

# --8<-- [start:load-arrests]
from sklego.datasets import load_arrests

df_arrests = load_arrests(as_frame=True)
df_arrests.head()
# --8<-- [end:load-arrests]

with open(_static_path / "arrests.md", "w") as f:
    f.write(df_arrests.head().to_markdown(index=False))

# --8<-- [start:plot-arrests]
X, y = load_arrests(return_X_y=True)

plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Distribution of released (target)")
# --8<-- [end:plot-arrests]

plt.savefig(_static_path / "arrests.png")
plt.clf()


######################################## Chicken #########################################
##########################################################################################

# --8<-- [start:load-chicken]
from sklego.datasets import load_chicken

df_chicken = load_chicken(as_frame=True)
df_chicken.head()
# --8<-- [end:load-chicken]

with open(_static_path / "chicken.md", "w") as f:
    f.write(df_chicken.head().to_markdown(index=False))

# --8<-- [start:plot-chicken]
X, y = load_chicken(return_X_y=True)

plt.hist(y)
plt.title("Distribution of weight (target)")
# --8<-- [end:plot-chicken]

plt.savefig(_static_path / "chicken.png")
plt.clf()


######################################## Hearts ##########################################
##########################################################################################

# --8<-- [start:load-hearts]
from sklego.datasets import load_hearts

df_hearts = load_hearts(as_frame=True)
df_hearts.head()
# --8<-- [end:load-hearts]

with open(_static_path / "hearts.md", "w") as f:
    f.write(df_hearts.head().to_markdown(index=False))

# --8<-- [start:plot-hearts]
X, y = load_hearts(return_X_y=True)

plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Distribution of presence of heart disease (target)")
# --8<-- [end:plot-hearts]

plt.savefig(_static_path / "hearts.png")
plt.clf()


######################################## Heroes ##########################################
##########################################################################################

# --8<-- [start:load-heroes]
from sklego.datasets import load_heroes

df_heroes = load_heroes(as_frame=True)
df_heroes.head()
# --8<-- [end:load-heroes]

with open(_static_path / "heroes.md", "w") as f:
    f.write(df_heroes.head().to_markdown(index=False))

# --8<-- [start:plot-heroes]
X, y = load_heroes(return_X_y=True)

plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Distribution of attack_type (target)")
# --8<-- [end:plot-heroes]

plt.savefig(_static_path / "heroes.png")
plt.clf()


######################################## Penguins ########################################
##########################################################################################

# --8<-- [start:load-penguins]
from sklego.datasets import load_penguins

df_penguins = load_penguins(as_frame=True)
df_penguins.head()
# --8<-- [end:load-penguins]

with open(_static_path / "penguins.md", "w") as f:
    f.write(df_penguins.head().to_markdown(index=False))

# --8<-- [start:plot-penguins]
X, y = load_penguins(return_X_y=True)

plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Distribution of species (target)")
# --8<-- [end:plot-penguins]

plt.savefig(_static_path / "penguins.png")
plt.clf()


###################################### Creditcards #######################################
##########################################################################################

# --8<-- [start:load-creditcards]
from sklego.datasets import fetch_creditcard

dict_creditcard = fetch_creditcard(as_frame=True)
df_creditcard = dict_creditcard["frame"]
df_creditcard.head()
# --8<-- [end:load-creditcards]

with open(_static_path / "creditcards.md", "w") as f:
    f.write(df_creditcard.head().to_markdown(index=False))

# --8<-- [start:plot-creditcards]
X, y = dict_creditcard["data"], dict_creditcard["target"]

plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Distribution of fraud (target)")
# --8<-- [end:plot-creditcards]

plt.savefig(_static_path / "creditcards.png")
plt.clf()

# --8<-- [start:load-ts]
from sklego.datasets import make_simpleseries

df_simpleseries = make_simpleseries(as_frame=True, n_samples=1500, trend=0.001)
df_simpleseries.head()
# --8<-- [end:load-ts]

with open(_static_path / "timeseries.md", "w") as f:
    f.write(df_simpleseries.head().to_markdown(index=True))

# --8<-- [start:plot-ts]
plt.plot(df_simpleseries["yt"])
plt.title("Timeseries yt")
# --8<-- [end:plot-ts]

plt.savefig(_static_path / "timeseries.png")
plt.clf()
