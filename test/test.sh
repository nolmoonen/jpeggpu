#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'usage: test.sh <jpeg_file> (optional: --write_out)'
    exit 0
fi

build_dir=$(dirname "$0")

# different subsampling factors
for css in 1x1 2x1 2x2 1x2 4x1 # 1x4 (nvJPEG does not support 1x4)
do
    name="$build_dir/$(basename "$1").$css.jpg"
    echo "creating tmp file $name.."
    convert $1 -sampling-factor $css $name
    $build_dir/jpeggpu_test $name $2
done

# grayscale
name_grayscale="$build_dir/$(basename "$1").grayscale.jpg"
echo "creating tmp file $name_grayscale.."
convert $1 -set colorspace Gray $name_grayscale
$build_dir/jpeggpu_test $name_grayscale $2
