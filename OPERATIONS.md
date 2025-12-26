# Operations & Deployment Guide

This document covers hardware integration, failure handling, and scaling procedures for the Edge AI Data Collector system on Raspberry Pi 5 with Hailo AI HAT.

## 1. Hardware Integration Checklist

### Phase 1: Physical Setup

- [ ] **Mount AI HAT**: Securely attach the Hailo-8/8L AI HAT to the Raspberry Pi 5 PCIe header.
- [ ] **Thermal Management**: Ensure the active cooler on the Pi 5 is functioning. The Hailo chip can get hot; ensure the HAT's thermal pad makes good contact.
- [ ] **Power Supply**: Use the official Raspberry Pi 27W USB-C Power Supply. The AI HAT draws additional power (up to 2.5W for 8L, more for 8).

### Phase 2: Driver & Software Installation

- [ ] **Install Hailo Drivers**:
  ```bash
  sudo apt install hailo-all
  ```
- [ ] **Verify Detection**:
  ```bash
  hailortcli scan
  # Should return: Hailo-8/8L PCIe device: 0000:01:00.0
  ```
- [ ] **Install Python Dependencies**:
  ```bash
  # It is recommended to use a virtual environment
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### Phase 2.1: C++ Implementation (Optional)

If you require lower latency, use the C++ version.

- **Install Build Dependencies**:
  ```bash
  sudo apt update
  sudo apt install -y build-essential cmake pkg-config libopencv-dev libyaml-cpp-dev libsqlite3-dev
  ```
- **Build**:
  ```bash
  cd src_cpp
  mkdir build && cd build
  cmake ..
  make -j4
  ```

### Phase 3: Model Compilation

- [ ] **Compile YOLOv8**: You cannot use `.pt` files directly. Compile to HEF using the [Hailo Dataflow Compiler (DFC)](https://hailo.ai/developer-zone/software/dataflow-compiler/).
  ```bash
  hailomz compile yolov8s-seg --hw-arch hailo8l --calib-path /path/to/calib_imgs/
  ```
- [ ] **Place Model**: Copy the `.hef` file to `models/yolov8s_seg.hef`.

---

## 2. Failure Handling & Troubleshooting

| Symptom                         | Probable Cause                   | Action                                                                                                                 |
| ------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **`HailoRTStatusException`**    | Driver mismatch or Overheating   | 1. Check `dmesg                                                                                                        | grep hailo`.<br>2. Ensure `hailortcli scan`works.<br>3. Check temperature:`vcgencmd measure_temp`. |
| **`CameraStream` Warning loop** | RTSP stream down / Network issue | 1. Verify camera IP is reachable via `ping`.<br>2. Check RTSP URL in VLC player.<br>3. Inspect switch/cable integrity. |
| **System Freezes**              | RAM exhaustion / Power sag       | 1. Enable ZRAM on Pi 5.<br>2. Verify 27W PSU usage.<br>3. Check logs for OOM Killer: `dmesg                            | grep -i kill`.                                                                                     |
| **Storage Full**                | Dataset filled SD card           | 1. Check disk usage: `df -h`.<br>2. Run cleanup script (see Scaling section).<br>3. Reduce capture `interval_seconds`. |

---

## 3. Automation & Scaling Procedures

### Step 1: Auto-Start Service (Systemd)

To ensure the collector runs 24/7 and restarts on crash:

1. Copy the service file:
   ```bash
   sudo cp deploy/datacollector.service /etc/systemd/system/
   ```
2. Edit paths if necessary:
   ```bash
   sudo nano /etc/systemd/system/datacollector.service
   ```
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable datacollector
   sudo systemctl start datacollector
   ```
4. Monitor logs:
   ```bash
   journalctl -u datacollector -f
   ```

### Step 2: Data Offloading (Cron + Rsync)

Don't let the SD card fill up. Set up a nightly job to move data to a NAS or Cloud.

Create `scripts/sync_data.sh`:

```bash
#!/bin/bash
# Move data older than 1 day to NAS
rsync -av --remove-source-files /home/pi/datacollector/dataset/ /mnt/nas/dataset/
# Prune empty directories
find /home/pi/datacollector/dataset/ -type d -empty -delete
```

Add to crontab (`crontab -e`):

```cron
0 3 * * * /bin/bash /home/pi/datacollector/scripts/sync_data.sh
```

### Step 3: Multi-Device Deployment (Ansible)

For scaling to 10+ Pis, do not manually configure each.

1. **Inventory File**: List all Pi IPs.
2. **Playbook**:
   - Install `hailo-all`.
   - `git clone` this repo.
   - `pip install`.
   - Copy `config.yaml` (template with unique camera IPs).
   - Enable systemd service.

### Step 4: Watchdog Configuration

Enable the Raspberry Pi hardware watchdog to reboot the system if the OS freezes completely.

1. Edit `/etc/systemd.conf`:
   ```
   RuntimeWatchdogSec=15
   RebootWatchdogSec=5m
   ```
2. The `datacollector` service handles software crashes (`Restart=always`), but the hardware watchdog handles kernel freezes.

---

## 4. Integration Todo List (Next Steps)

- [ ] **Validation**: Run the system for 24h continuously to check for memory leaks.
- [ ] **Network Storage**: Mount a NAS via NFS/SMB for long-term storage if local SD is small.
- [ ] **Time Sync**: Ensure NTP is active (`timedatectl status`). Timestamps are critical for dataset validity.
- [ ] **LED Status**: (Optional) Add GPIO control to flash an LED when recording is active for visual feedback.
