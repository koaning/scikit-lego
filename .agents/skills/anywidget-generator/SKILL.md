---
name: anywidget-generator
description: Generate anywidget components for marimo notebooks.
---

When writing an anywidget use vanilla javascript in `_esm` and do not forget about `_css`. The css should look bespoke in light mode and dark mode. Keep the css small unless explicitly asked to go the extra mile. When you display the widget it must be wrapped via `widget = mo.ui.anywidget(OriginalAnywidget())`. You can also point `_esm` and `_css` to external files if needed using pathlib. This makes sense if the widget does a lot of elaborate JavaScript or CSS.

<example title="Example of simple anywidget implementation">
import anywidget
import traitlets


class CounterWidget(anywidget.AnyWidget):
    _esm = """
    // Define the main render function
    function render({ model, el }) {
      let count = () => model.get("number");
      let btn = document.createElement("b8utton");
      btn.innerHTML = `count is ${count()}`;
      btn.addEventListener("click", () => {
        model.set("number", count() + 1);
        model.save_changes();
      });
      model.on("change:number", () => {
        btn.innerHTML = `count is ${count()}`;
      });
      el.appendChild(btn);
    }
    // Important! We must export at the bottom here!
    export default { render };
    """
    _css = """button{
      font-size: 14px;
    }"""
    number = traitlets.Int(0).tag(sync=True)

widget = mo.ui.anywidget(CounterWidget())
widget

# Grabbing the widget from another cell, `.value` is a dictionary.
print(widget.value["number"])
</example>

The above is a minimal example that could work for a simple counter widget. In general the widget can become much larger because of all the JavaScript and CSS required. Unless the widget is dead simple, you should consider using external files for `_esm` and `_css` using pathlib. 

When sharing the anywidget, keep the example minimal. No need to combine it with marimo ui elements unless explicitly stated to do so.

## Best Practices

Unless specifically told otherwise, assume the following:

1. **Use vanilla JavaScript in `_esm`**:
   - Define a `render` function that takes `{ model, el }` as parameters
   - Use `model.get()` to read trait values
   - Use `model.set()` and `model.save_changes()` to update traits
   - Listen to changes with `model.on("change:traitname", callback)`
   - Export default with `export default { render };` at the bottom
   - All widgets inherit from `anywidget.AnyWidget`, so `widget.observe(handler)`
     remains the standard way to react to state changes.
   - Python constructors tend to validate bounds, lengths, or choice counts; let the
     raised `ValueError/TraitError` guide you instead of duplicating the logic.

2. **Include `_css` styling**:
   - Keep CSS minimal unless explicitly asked for more
   - Make it look bespoke in both light and dark mode
   - Use CSS media query for dark mode: `@media (prefers-color-scheme: dark) { ... }`

3. **Wrap the widget for display**:
   - Always wrap with marimo: `widget = mo.ui.anywidget(OriginalAnywidget())`
   - Access values via `widget.value` which returns a dictionary

4. **Keep examples minimal**:
   - Add a marimo notebook that highlights the core utility
   - Show basic usage only
   - Don't combine with other marimo UI elements unless explicitly requested

Dumber is better. Prefer obvious, direct code over clever abstractions—someone
new to the project should be able to read the code top-to-bottom and grok it
without needing to look up framework magic or trace through indirection.
