from __future__ import annotations

from pathlib import Path
from typing import Final

from sklego.this import poem

DESTINATION_PATH: Final[Path] = Path("docs") / "this.md"

content = f"""
# Import This

In Python there's a poem that you can read by importing the `this` module.

```py
import this
```

It has wonderful lessons that the authors of the language learned while designing the python language.

In the same tradition we've done the same thing. Folks who have made significant contributions have also been asked to
contribute to the poem.

You can read it via:

```py
from sklego import this
```

```console
{poem}
```
"""

with DESTINATION_PATH.open(mode="w") as destination:
    destination.write(content)
