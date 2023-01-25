#!/bin/bash

set -x

update() {
    target_path="$1"
    url="$2"
    
    if [ -f "${target_path}" -o -d "${target_path}" ]
    then
        echo "${target_path} ... UP TO DATE."
    else                
        tar_file=${url##*/}
        curl "${url}" --output "${tar_file}"
        tar -xvf "${tar_file}"
        rm "${tar_file}"
    fi
}

# download big test data
#update large_model.msh https://flow.nti.tul.cz/endorse_large_data/large_model.tar.gz
update large_model_local.msh2 http://flow.nti.tul.cz/endorse_large_data/large_model_local.tar.gz 

#update flow_fields.pvd https://flow.nti.tul.cz/endorse_large_data/output_transport_2022_5.tar.gz
#update solute_fields.pvd https://flow.nti.tul.cz/endorse_large_data/output_transport_2022_5.tar.gz
