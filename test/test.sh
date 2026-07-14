#!/bin/bash

# Copyright (c) 2024 Nol Moonen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https:#www.gnu.org/licenses/>.

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
