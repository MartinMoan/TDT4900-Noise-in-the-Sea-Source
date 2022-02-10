#!/usr/bin/env python3
import os, shutil
import pathlib
from rich import print

def get_install_location():
    if os.name == "posix":
        return pathlib.Path("/usr/local/bin")
    else:
        raise Exception("The current OS is not supported")

def main():
    raw_script_path = pathlib.Path(__file__).parent.joinpath("play.py")
    new_script_name = raw_script_path.stem
    install_location = get_install_location()
    new_script_isntall_path = install_location.joinpath(new_script_name)
    print(f"Installing script...")
    installed_path = shutil.copy(raw_script_path, new_script_isntall_path)
    print(f"Script successfully installed to {installed_path}")

if __name__ == "__main__":
    main()