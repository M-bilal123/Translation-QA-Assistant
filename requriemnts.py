import importlib
import pkg_resources

# List of required libraries
required_libs = [
    "streamlit",
    "pandas",
    "pathlib",
    "sys",
    "datetime",
    "io"
]

print("🔍 Checking installed libraries and versions...\n")

for lib in required_libs:
    try:
        module = importlib.import_module(lib)
        # Some built-in modules (like pathlib, sys, datetime, io) don't have __version__
        version = getattr(module, "__version__", "Built-in module (no version)")
        print(f"✅ {lib} - {version}")
    except ImportError:
        print(f"❌ {lib} is missing. Install it with: pip install {lib}")

print("\n✅ Check complete!")


# streamlit: 1.50.0
# pandas: 2.0.3
# openpyxl: 3.0.10
# reportlab: 4.4.4