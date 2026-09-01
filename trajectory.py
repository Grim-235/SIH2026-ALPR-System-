import argparse
import math
import sqlite3
from typing import List, Dict, Optional, Any

import folium
from folium import plugins

try:
    from alpr.database import (
        init_db,
        query_plate_history,
        get_camera_heatmap_data,
        get_top_routes
    )
except ImportError:
    # Fallback/mock for standalone testing if needed
    pass

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0  # Earth radius in kilometers

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def estimate_speed(lat1: float, lon1: float, lat2: float, lon2: float, time_delta_seconds: float) -> Optional[float]:
    """Calculate approximate speed in km/h using haversine distance between two GPS points."""
    if time_delta_seconds <= 0:
        return None
    
    distance_km = _haversine_distance(lat1, lon1, lat2, lon2)
    speed_kmh = (distance_km / time_delta_seconds) * 3600.0
    return speed_kmh

def get_vehicle_trajectory(conn: sqlite3.Connection, plate_text: str) -> List[Dict[str, Any]]:
    """Query all detections for the plate ordered by timestamp."""
    history = query_plate_history(conn, plate_text)
    
    trajectory = []
    for item in history:
        trajectory.append({
            'camera_name': item.get('camera_name'),
            'latitude': item.get('latitude'),
            'longitude': item.get('longitude'),
            'timestamp': item.get('timestamp'),
            'camera_id': item.get('camera_id')
        })
        
    return trajectory

def _get_popup_html(title: str, content: Dict[str, Any]) -> str:
    """Generate styled HTML for Folium popups using a dark theme."""
    html = f'''
    <div style="background-color: #212121; color: #ffffff; padding: 12px; border-radius: 8px; font-family: sans-serif; min-width: 180px;">
        <h4 style="margin: 0 0 10px 0; color: #00e5ff; border-bottom: 1px solid #444; padding-bottom: 6px; font-size: 16px;">{title}</h4>
    '''
    for k, v in content.items():
        html += f'<p style="margin: 6px 0; font-size: 13px;"><b style="color: #aaaaaa;">{k}:</b> {v}</p>'
    html += '</div>'
    return html

def generate_trajectory_map(conn: sqlite3.Connection, plate_text: str, output_html: Optional[str] = None) -> Optional[folium.Map]:
    """Create an interactive Folium map showing a vehicle's trajectory."""
    trajectory = get_vehicle_trajectory(conn, plate_text)
    
    valid_points = [t for t in trajectory if t.get('latitude') is not None and t.get('longitude') is not None]
    if not valid_points:
        return None
        
    avg_lat = sum(t['latitude'] for t in valid_points) / len(valid_points)
    avg_lon = sum(t['longitude'] for t in valid_points) / len(valid_points)
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles='CartoDB dark_matter')
    
    # Add title
    title_html = f'''
         <h3 align="center" style="font-size:22px; color: #00e5ff; margin-top: 15px; font-family: sans-serif; font-weight: bold; text-shadow: 0 0 5px rgba(0,229,255,0.5);">
         Vehicle Trajectory: {plate_text}
         </h3>
         '''
    m.get_root().html.add_child(folium.Element(title_html))

    coordinates = []
    
    for idx, point in enumerate(trajectory):
        lat = point['latitude']
        lon = point['longitude']
        if lat is None or lon is None:
            continue
            
        coordinates.append((lat, lon))
        visit_order = idx + 1
        
        popup_content = {
            'Timestamp': point['timestamp'],
            'Order': visit_order,
            'Camera ID': point['camera_id']
        }
        
        popup_iframe = folium.IFrame(html=_get_popup_html(point['camera_name'] or f"Camera {point['camera_id']}", popup_content), width=260, height=160)
        popup = folium.Popup(popup_iframe, max_width=260)
        
        icon = folium.DivIcon(
            html=f"""
                <div style="
                    background-color: #00e5ff;
                    color: #000000;
                    border-radius: 50%;
                    width: 28px;
                    height: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 14px;
                    border: 2px solid #ffffff;
                    box-shadow: 0 0 8px rgba(0,229,255,0.9);
                ">
                    {visit_order}
                </div>
            """
        )
        
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=icon
        ).add_to(m)
        
    # Draw polyline connecting points
    if len(coordinates) > 1:
        folium.PolyLine(
            locations=coordinates,
            color='#00e5ff',
            weight=3,
            dash_array='8, 8',
            opacity=0.9
        ).add_to(m)
        
        # Add AntPath for directional flow
        try:
            plugins.AntPath(
                locations=coordinates,
                color="#ffffff",
                pulse_color="#00e5ff",
                weight=3,
                delay=800,
                dash_array=[10, 20]
            ).add_to(m)
        except Exception:
            pass # Fallback if AntPath is unavailable
        
    if output_html:
        m.save(output_html)
        print(f"Trajectory map saved to {output_html}")
        
    return m

def generate_overview_map(conn: sqlite3.Connection, output_html: Optional[str] = None) -> folium.Map:
    """Create an overview map showing all camera locations with detection counts and routes."""
    heatmap_data = get_camera_heatmap_data(conn)
    top_routes = get_top_routes(conn, limit=100)
    
    valid_cams = [c for c in heatmap_data if c.get('latitude') is not None and c.get('longitude') is not None] if heatmap_data else []
    if valid_cams:
        avg_lat = sum(c['latitude'] for c in valid_cams) / len(valid_cams)
        avg_lon = sum(c['longitude'] for c in valid_cams) / len(valid_cams)
    else:
        avg_lat, avg_lon = 20.5937, 78.9629 # Default center (India)
        
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles='CartoDB dark_matter')
    
    title_html = '''
         <h3 align="center" style="font-size:22px; color: #ff9800; margin-top: 15px; font-family: sans-serif; font-weight: bold; text-shadow: 0 0 5px rgba(255,152,0,0.5);">
         City Overview & Traffic Routes
         </h3>
         '''
    m.get_root().html.add_child(folium.Element(title_html))

    cameras_layer = folium.FeatureGroup(name='Cameras', show=True)
    routes_layer = folium.FeatureGroup(name='Routes', show=True)
    
    camera_locations = {}
    max_count = max([c.get('count', 1) for c in heatmap_data]) if heatmap_data else 1
    
    for cam in heatmap_data:
        lat, lon = cam.get('latitude'), cam.get('longitude')
        if lat is None or lon is None:
            continue
            
        cam_id = cam.get('camera_id')
        camera_locations[cam_id] = (lat, lon)
        count = cam.get('count', 0)
        
        radius = max(6, min(25, (count / max_count) * 25))
        
        # Query top 3 plates for this camera
        top_plates = []
        try:
            cur = conn.cursor()
            cur.execute('''
                SELECT plate_text, COUNT(*) as c FROM detections 
                WHERE camera_id = ? 
                GROUP BY plate_text 
                ORDER BY c DESC LIMIT 3
            ''', (cam_id,))
            top_plates = [row[0] for row in cur.fetchall()]
        except Exception:
            pass
            
        popup_content = {
            'Total Detections': count,
            'Top Plates': ', '.join(top_plates) if top_plates else 'N/A'
        }
        
        cam_name = cam.get('name') or f"Camera {cam_id}"
        popup_iframe = folium.IFrame(html=_get_popup_html(cam_name, popup_content), width=260, height=150)
        popup = folium.Popup(popup_iframe, max_width=260)
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color='#00e5ff',
            weight=2,
            fill=True,
            fill_color='#00e5ff',
            fill_opacity=0.6,
            popup=popup,
            tooltip=f"{cam_name} (Hits: {count})"
        ).add_to(cameras_layer)
        
    # Draw route lines
    max_route_count = max([r.get('count', 1) for r in top_routes]) if top_routes else 1
    
    for route in top_routes:
        from_cam = route.get('from_camera')
        to_cam = route.get('to_camera')
        
        if from_cam in camera_locations and to_cam in camera_locations:
            count = route.get('count', 1)
            weight = max(1, min(12, (count / max_route_count) * 12))
            
            p1 = camera_locations[from_cam]
            p2 = camera_locations[to_cam]
            
            folium.PolyLine(
                locations=[p1, p2],
                color='#ff9800',
                weight=weight,
                opacity=0.5,
                tooltip=f"Route: {route.get('from_name', from_cam)} -> {route.get('to_name', to_cam)} (Trips: {count})"
            ).add_to(routes_layer)
            
    cameras_layer.add_to(m)
    routes_layer.add_to(m)
    folium.LayerControl().add_to(m)
    
    if output_html:
        m.save(output_html)
        print(f"Overview map saved to {output_html}")
        
    return m

def main():
    parser = argparse.ArgumentParser(description="Generate ANPR Maps with Folium")
    parser.add_argument("--plate", type=str, help="Plate number to trace (trajectory map)")
    parser.add_argument("--overview", action="store_true", help="Generate overview map of all cameras and routes")
    parser.add_argument("--db", type=str, default="data/alpr.db", help="Path to SQLite DB (default: data/alpr.db)")
    parser.add_argument("--output", type=str, help="Output HTML file path")
    
    args = parser.parse_args()
    
    if not args.plate and not args.overview:
        parser.error("You must specify either --plate or --overview")
        
    try:
        conn = init_db(args.db)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    try:
        if args.overview:
            output_path = args.output or "overview_map.html"
            generate_overview_map(conn, output_html=output_path)
        elif args.plate:
            output_path = args.output or f"trajectory_{args.plate}.html"
            generate_trajectory_map(conn, args.plate, output_html=output_path)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
