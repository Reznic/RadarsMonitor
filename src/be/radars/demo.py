from radar_tracks_server import RadarTracksServer
import time  
import math


if __name__ == "__main__":
   
   # Create server instance
   server = RadarTracksServer(port=1337)
   
   # Start the server
   server.start_server()
   
   # Initialize radar status
   server.update_radar_status(
       radar_id="radar1",
       is_active=True,
       orientation_angle=45.0
   )
   
   # Simulate a track moving closer
   initial_range = 100.0  # Start at 100m
   final_range = 5.0     # End at 5m
   samples = 200
   
   # Calculate range decrease per step
   range_step = (initial_range - final_range) / samples
   
   # Simulate small azimuth oscillation using sine wave
   base_azimuth = 45.0
   
   try:
       # Generate 200 samples
       for i in range(samples):
           # Calculate current range
           current_range = initial_range - (range_step * i)
           
           # Add small sine wave variation to azimuth (-5 to +5 degrees)
           azimuth_variation = 5 * math.sin(i * 2 * math.pi / 50)  # Complete oscillation every 50 samples
           current_azimuth = base_azimuth + azimuth_variation
           
           # Ensure azimuth stays in 0-359 range
           current_azimuth = current_azimuth % 360
           
           # Update track data
           server.update_radar_data(
               radar_id="radar1",
               track_id=1,
               azimuth=current_azimuth,
               range_meters=current_range
           )
           
           # Simulate radar status changes
           # Radar orientation follows a slower oscillation pattern
           orientation_angle = 70
           
           # Simulate radar going active/inactive every 50 samples
           is_active = (i % 50) < 40  # Active for 40 samples, inactive for 10 samples
           
           # Update radar status
           server.update_radar_status(
               radar_id="radar1",
               is_active=is_active,
               orientation_angle=orientation_angle   # Keep angle in 0-359 range
           )
           
           # Add a second radar with different pattern
           second_orientation = 190 
           server.update_radar_status(
               radar_id="radar2",
               is_active=not is_active,  # Opposite active status of radar1
               orientation_angle=second_orientation 
           )
           
           # Sleep for a short time to simulate real-time updates
           time.sleep(3)  # Update every 3 seconds
           
           # Print update (optional)
           print(f"Sample {i+1}/200:")
           print(f"  Radar1: Active={is_active}, Orientation={orientation_angle:.1f}°")
           print(f"  Radar2: Active={not is_active}, Orientation={second_orientation:.1f}°")
           
       print("Simulation completed")
       
       # Keep server running
       while True:
           time.sleep(1)
           
   except KeyboardInterrupt:
       server.stop_server()
