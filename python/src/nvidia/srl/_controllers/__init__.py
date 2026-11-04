# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""nvidia-srl-controllers distribution version info."""

DISTRIBUTION_NAME = "nvidia_srl_controllers"

# If true, disables printing the warning message when the fallback version is used.
NO_WARN = True


# NOTE (roflaherty): This is inspired by how matplotlib generates its version value.
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L161
def _get_version() -> str:
    """Return the version string used for __version__."""
    # Standard Library
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _get_fallback_version(no_warn=NO_WARN)


def _get_fallback_version(no_warn: bool = False) -> str:
    """Return an arbitrary version value and print a user warning.

    Args:
        no_warn: If true, the warning message is not printed.
    """
    # Standard Library
    import traceback
    import warnings
    from typing import Optional, Type, Union

    # Arbitrary version value
    this_version = "0.0.0.dev0"

    def warning_on_one_line(
        message: Union[Warning, str],
        category: Type[Warning],
        filename: str,
        lineno: int,
        line: Optional[str] = None,
    ) -> str:
        return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)

    warnings.formatwarning = warning_on_one_line
    warn_msg = (
        "The following error was caught while trying to obtain the version value for the"
        f" '{DISTRIBUTION_NAME}' distribution. The version value will be set to the arbitrary"
        f" value of '{this_version}'"
    )

    if not no_warn:
        warnings.warn(warn_msg)
        traceback.print_exc()

    return this_version
