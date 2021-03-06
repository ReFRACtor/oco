#!/bin/sh

# Find GNU version of utils
# Prefixed with g* on Mac OSX
# Install coreutils package of Homebrew or MacPorts

if [ -n "$(which greadlink)" ]; then
    readlink_bin=$(which greadlink)
else
    readlink_bin=$(which readlink)
fi

if [ ! -e "$readlink_bin" ]; then
    echo "$(basename $0): readlink binary does not exist: $readlink_bin" >&2
    exit 1
fi

if [ -n "$(which gpaste)" ]; then
    paste_bin=$(which gpaste)
else
    paste_bin=$(which paste)
fi

if [ ! -e "$paste_bin" ]; then
    echo "$(basename $0): paste binary does not exist: $paste_bin" >&2
    exit 1
fi

interface_dir=$(dirname $0)

# Extracts swig interface filenames depended on by the passed interface filename
for dep_fn in $(grep -E '^\s*%(include|import)' $1 | sed -E 's/^.*"([^"]*)".*$/\1/'); do
    if [ -e $interface_dir/$dep_fn ]; then
        echo $(${readlink_bin} -f $interface_dir/$dep_fn)
        # For files present in this directory, parse out their dependencies recursively
        $0 $interface_dir/$dep_fn | tr ';' '\n'
    elif [ -e $interface_dir/../lib/$dep_fn ]; then
        echo $(${readlink_bin} -f $interface_dir/../lib/$dep_fn)
    fi
done | sort | uniq | ${paste_bin} -s -d ';' 
