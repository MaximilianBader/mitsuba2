# This script adds Mitsuba and the Python bindings to the shell's search path.
# It must be executed via the 'source' command so that it can modify the
# relevant environment variables.

MITSUBA_DIR=""

if [ "$BASH_VERSION" ]; then
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        MITSUBA_DIR=$(dirname "$BASH_SOURCE")
        MITSUBA_DIR=$(builtin cd "$MITSUBA_DIR"; builtin pwd)
    fi
elif [ "$ZSH_VERSION" ]; then
    if [[ -n ${(M)zsh_eval_context:#file} ]]; then
        MITSUBA_DIR=$(dirname "$0:A")
    fi
fi

if [ -z "$MITSUBA_DIR" ]; then
    echo "This script must be executed via the 'source' command, i.e.:"
    echo "$ source ${0}"
    exit 0
fi

echo $MITSUBA_DIR

export PYTHONPATH="$MITSUBA_DIR/python:$PYTHONPATH"
export PATH="$MITSUBA_DIR:$PATH"
