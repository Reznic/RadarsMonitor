pip install --no-index --find-links=./packages -r requirements.txt
sudo chmod -R 777 ./*.*
sudo cp radar_adapter.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable radar_adapter.service
sudo systemctl start radar_adapter.service

