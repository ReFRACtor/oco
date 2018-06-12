#!/bin/sh

interface_dir=$(dirname $0)

# Extracts swig interface filenames depended on by the passed interface filename
for dep_fn in $(grep -E '^\s*%(include|import)' $1 | sed -E 's/^.*"([^"]*)".*$/\1/'); do
    if [ -e $interface_dir/$dep_fn ]; then
        echo $(readlink -f $interface_dir/$dep_fn)
        # For files present in this directory, parse out their dependencies recursively
        $0 $interface_dir/$dep_fn | tr ';' '\n'
    elif [ -e $interface_dir/../lib/$dep_fn ]; then
        echo $(readlink -f $interface_dir/../lib/$dep_fn)
    fi
done | sort | uniq | paste -s -d ';' 
