
ssh fptxurdinaga.in <<EOF
    docker cp wordpress:/var/www/html/wp-content/uploads/wp_dndcf7_uploads/wpcf7-files aldi_baterako_argazkiak_eramateko
EOF
scp -r fptxurdinaga.in:aldi_baterako_argazkiak_eramateko/* ~/multzoa_ekarritakoa/
ssh fptxurdinaga.in rm -r aldi_baterako_argazkiak_eramateko
