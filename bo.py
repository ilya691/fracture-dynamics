import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse

# --- Классы ---
class MaterialPoint:
    def __init__(self, x, y, mass=1.0, is_fixed=False, material='concrete'):
        self.mass = mass
        self.is_fixed = is_fixed
        self.pos = np.array([x, y], dtype=float)
        self.prev_pos = np.array([x, y], dtype=float)
        self.acc = np.array([0.0, 0.0], dtype=float)
        self.material = material
        self.damage = 0.0 

class Link:
    def __init__(self, p1_idx, p2_idx, rest_length, max_strain, stiffness, material_type):
        self.p1 = p1_idx
        self.p2 = p2_idx
        self.rest_length = rest_length
        self.max_strain = max_strain 
        self.stiffness = stiffness
        self.material_type = material_type
        self.broken = False

# --- Создание бетонной плиты ---
def create_concrete_wall(cols, rows, spacing, start_x, start_y, params):
    points, links = [], []
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * spacing
            y = start_y + r * spacing
            is_fixed = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
            points.append(MaterialPoint(x, y, mass=params['concrete_density'], is_fixed=is_fixed, material='concrete'))
            
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c < cols - 1:
                links.append(Link(idx, idx + 1, spacing, params['concrete_max_strain'], params['concrete_stiffness'], 'concrete_link'))
            if r < rows - 1:
                links.append(Link(idx, idx + cols, spacing, params['concrete_max_strain'], params['concrete_stiffness'], 'concrete_link'))
            if c < cols - 1 and r < rows - 1:
                diag_len = spacing * np.sqrt(2)
                links.append(Link(idx, idx + cols + 1, diag_len, params['concrete_max_strain'], params['concrete_stiffness'] * 0.5, 'concrete_link'))
                links.append(Link(idx + 1, idx + cols, diag_len, params['concrete_max_strain'], params['concrete_stiffness'] * 0.5, 'concrete_link'))
    return points, links

# --- Создание овального чугунного ударника ---
def create_oval_cast_iron(radius_x, radius_y, spacing, center_x, center_y, params):
    points, links = [], []
    for y in np.arange(center_y - radius_y, center_y + radius_y, spacing):
        for x in np.arange(center_x - radius_x, center_x + radius_x, spacing):
            if ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1.0:
                points.append(MaterialPoint(x, y, mass=params['iron_density'], is_fixed=False, material='iron'))

    max_dist = spacing * 1.5
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i].pos - points[j].pos)
            if dist <= max_dist:
                links.append(Link(i, j, dist, params['iron_max_strain'], params['iron_stiffness'], 'iron_link'))
                
    return points, links

# --- Физика ---
def compute_forces(points_wall, links_wall, points_proj, links_proj):
    for p in points_wall + points_proj:
        p.acc = np.array([0.0, 0.0])
        
    # Бетон
    for link in links_wall:
        if link.broken: continue
        p1, p2 = points_wall[link.p1], points_wall[link.p2]
        delta = p2.pos - p1.pos
        dist = np.linalg.norm(delta)
        if dist == 0: continue
        strain = (dist - link.rest_length) / link.rest_length
        
        if abs(strain) > link.max_strain:
            damage_inc = (abs(strain) - link.max_strain) * 25.0 
            p1.damage = min(1.0, p1.damage + damage_inc)
            p2.damage = min(1.0, p2.damage + damage_inc)
            if p1.damage >= 1.0 or p2.damage >= 1.0:
                link.broken = True
                continue
        
        cur_stiff = link.stiffness * (1.0 - max(p1.damage, p2.damage) * 0.9)
        force = (cur_stiff * (dist - link.rest_length)) * (delta / dist)
        p1.acc += force / p1.mass
        p2.acc -= force / p2.mass

    # Чугун (хрупкий)
    for link in links_proj:
        if link.broken: continue
        p1, p2 = points_proj[link.p1], points_proj[link.p2]
        delta = p2.pos - p1.pos
        dist = np.linalg.norm(delta)
        if dist == 0: continue
        strain = (dist - link.rest_length) / link.rest_length
        
        if strain > link.max_strain: # Чугун не тянется
            link.broken = True
            continue
                
        force = (link.stiffness * (dist - link.rest_length)) * (delta / dist)
        p1.acc += force / p1.mass
        p2.acc -= force / p2.mass

    # Контакт
    contact_radius = 6.0
    for pw in points_wall:
        if pw.is_fixed: continue
        for pp in points_proj:
            delta = pw.pos - pp.pos
            dist = np.linalg.norm(delta)
            if dist < contact_radius:
                force_mag = 50000.0 * (1.0 - pw.damage * 0.8) * (contact_radius - dist)
                force = force_mag * (delta / dist if dist > 0 else np.array([1.0, 0.0]))
                pw.acc += force / pw.mass
                pp.acc -= force / pp.mass
                if force_mag > 5000:
                    pw.damage = min(1.0, pw.damage + 0.05)

def integrate_verlet(points, dt):
    for p in points:
        if p.is_fixed: continue
        new_pos = 2 * p.pos - p.prev_pos + (dt**2) * p.acc
        p.prev_pos = p.pos.copy()
        p.pos = new_pos

def main():
    parser = argparse.ArgumentParser(description="Удар овальным чугунным ударником по бетону")
    parser.add_argument('--speed', type=float, default=90.0, help='Скорость ударника')
    parser.add_argument('--frames', type=int, default=200, help='Количество кадров')
    parser.add_argument('--output', type=str, default='cast_iron_impact.gif', help='Имя файла')
    
    # === ДОБАВЛЕНЫ АРГУМЕНТЫ ДЛЯ БЕТОНА ===
    parser.add_argument('--strength', type=float, default=1.0, help='Множитель прочности бетона')
    parser.add_argument('--stiffness', type=float, default=1.0, help='Множитель жесткости бетона')
    
    args = parser.parse_args()

    # === ИЗМЕНЕНЫ ПАРАМЕТРЫ БЕТОНА С УЧЕТОМ АРГУМЕНТОВ ===
    params = {
        'concrete_density': 2.4,        
        'concrete_stiffness': 5000.0 * args.stiffness,   
        'concrete_max_strain': 0.015 * args.strength,    
        
        'iron_density': 7.8,            
        'iron_stiffness': 25000.0,      
        'iron_max_strain': 0.02         
    }

    print(f"--- Параметры удара ---")
    print(f"Скорость: {args.speed} | Кадров: {args.frames}")
    print(f"Жесткость бетона: {params['concrete_stiffness']:.1f} (x{args.stiffness})")
    print(f"Прочность бетона: {params['concrete_max_strain']:.4f} (x{args.strength})")
    
    points_wall, links_wall = create_concrete_wall(20, 40, 5.0, 250, 50, params)
    points_proj, links_proj = create_oval_cast_iron(
        radius_x=25, radius_y=40, spacing=4.0, center_x=100, center_y=150, params=params
    )

    for p in points_proj:
        p.prev_pos[0] = p.pos[0] - args.speed * 0.002

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 300)
    ax.set_facecolor('black')
    ax.set_xticks([]); ax.set_yticks([])

    scatter_wall = ax.scatter([], [], s=4)
    scatter_proj = ax.scatter([], [], c='#87CEEB', s=8)
    
    lines_wall = [ax.plot([], [], color='gray', linewidth=0.3)[0] for _ in links_wall]
    lines_proj = [ax.plot([], [], color='#4682B4', linewidth=0.8)[0] for _ in links_proj]

    dt = 0.002

    def update(frame):
        for _ in range(5):
            compute_forces(points_wall, links_wall, points_proj, links_proj)
            integrate_verlet(points_wall, dt)
            integrate_verlet(points_proj, dt)

        wall_colors = [(0.5 + p.damage * 0.5, 0.5 * (1 - p.damage), 0.5 * (1 - p.damage)) for p in points_wall]
        scatter_wall.set_offsets(np.array([p.pos for p in points_wall]))
        scatter_wall.set_color(wall_colors)
        
        for i, link in enumerate(links_wall):
            if link.broken: 
                lines_wall[i].set_data([], [])
            else:
                p1, p2 = points_wall[link.p1].pos, points_wall[link.p2].pos
                lines_wall[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        scatter_proj.set_offsets(np.array([p.pos for p in points_proj]))
        for i, link in enumerate(links_proj):
            if link.broken: 
                lines_proj[i].set_data([], [])
            else:
                p1, p2 = points_proj[link.p1].pos, points_proj[link.p2].pos
                lines_proj[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        if frame == args.frames - 1:
            broken_wall = sum(1 for l in links_wall if l.broken)
            total_wall = len(links_wall)
            broken_proj = sum(1 for l in links_proj if l.broken)
            total_proj = len(links_proj)
            print(f"\n--- Итоги ---")
            print(f"Разрушено связей бетона: {broken_wall}/{total_wall} ({(broken_wall/total_wall)*100:.1f}%)")
            print(f"Разрушено связей чугуна: {broken_proj}/{total_proj} ({(broken_proj/total_proj)*100:.1f}%)")
            print(f"Конечная координата X ударника: {max(p.pos[0] for p in points_proj):.1f}")

        if frame % 50 == 0: 
            print(f"Кадр {frame}/{args.frames}...")
        return [scatter_wall, scatter_proj] + lines_wall + lines_proj

    print("Начинаем рендеринг GIF...")
    anim = FuncAnimation(fig, update, frames=args.frames, interval=20, blit=True)
    anim.save(args.output, writer=PillowWriter(fps=30))
    print(f"Готово! Файл '{args.output}' сохранен.")
    plt.close()

if __name__ == "__main__":
    main()