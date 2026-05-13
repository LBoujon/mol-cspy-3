#!/bin/bash
USER_SERVICE_DIR="/etc/systemd/user"
SERVICE_NAME="cspy_scheduler_daemon"
systemctl --user stop "${SERVICE_NAME}"
echo "Copying ${SERVICE_NAME}.service -> ${USER_SERVICE_DIR}"
sudo cp "${SERVICE_NAME}.service" "${USER_SERVICE_DIR}"
echo "Reloading systemd unit list"
systemctl --user daemon-reload
echo "Starting ${SERVICE_NAME} service"
systemctl --user start "${SERVICE_NAME}"
echo "Enabling automatic start of ${SERVICE_NAME}"
systemctl --user enable "${SERVICE_NAME}"
# If you wish to make your systemd instance independent from your user sessions
# i.e. so that the service starts at boot time even if you don't login
# and keeps running until a shutdown/reboot run the following
# sudo loginctl enable-linger $USER
