for deg in 20 40; do
  for i in {0..29}; do
    src="../gns-reptile/cbgeo/datasets/indiv_traj/${deg}_deg_${i}"
    dest="/work/10114/naveen_raj_manoharan/frontera/reptile_dataset/${deg}_deg_${i}"
    if [ -d "$src" ]; then
      echo "Copying $src to $dest"
      cp -r "$src" "$dest"
    else
      echo "Source directory $src does not exist. Skipping."
    fi
  done
done