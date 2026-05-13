#!/bin/bash
output=${1%.db}.rebuild.db
echo "Recreating database $1 -> ${output}"
for table_name in crystal descriptor trial_structure; do
    echo "Rebuilding table: $table_name"
    sqlite3 ${1} ".dump $table_name" | sqlite3 $output
done
backup=${1%.db}.corrupt
echo "Rename $1 -> ${backup}"
mv ${1} ${backup}
echo "Rename ${output} -> ${1}"
mv ${output} ${1}
