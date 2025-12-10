# Debug breakpoints tests are run via pytest in test_debug.py
# This file validates the NuGet package was created

import os

print("Checking debug breakpoints test output...")

# Check NuGet package
uipath_dir = ".uipath"
assert os.path.exists(uipath_dir), "NuGet package directory (.uipath) not found"

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith('.nupkg')]
assert nupkg_files, "NuGet package file (.nupkg) not found in .uipath directory"

print(f"NuGet package found: {nupkg_files[0]}")
print("Debug breakpoints tests completed via pytest.")
