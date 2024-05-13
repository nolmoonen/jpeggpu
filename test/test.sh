#!/bin/bash

# Copyright (c) 2024 Nol Moonen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
