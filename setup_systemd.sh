#!/bin/bash
# Setup script to configure facerecWeb systemd service with correct permissions

echo "🔧 Setting up facerecWeb systemd service..."

# Stop any running gunicorn processes
echo "1. Stopping current gunicorn processes..."
pkill -f gunicorn
sleep 2

# Copy the updated service file
echo "2. Installing systemd service file..."
sudo cp /tmp/facerecweb.service /etc/systemd/system/facerecweb.service

# Reload systemd
echo "3. Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service to start on boot
echo "4. Enabling service to start on boot..."
sudo systemctl enable facerecweb

# Start the service
echo "5. Starting facerecWeb service..."
sudo systemctl start facerecweb

# Wait a moment for service to start
sleep 3

# Check status
echo ""
echo "✅ Setup complete! Service status:"
sudo systemctl status facerecweb --no-pager -l

echo ""
echo "📊 To manage the service:"
echo "  sudo systemctl start facerecweb    # Start service"
echo "  sudo systemctl stop facerecweb     # Stop service"
echo "  sudo systemctl restart facerecweb  # Restart service"
echo "  sudo systemctl status facerecweb   # Check status"
