# SCRIPT PEMOTONGAN YANG GAMBARNYA TERSIMPAN DI DOWNLOAD


# usage: python -u line2hist.py <inputimage>
import os
#os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
from skimage.morphology import skeletonize
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)
def freeman(x, y):
    if (y==0):
        y=1e-9 # so that we escape the divby0 exception
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (abs(x/y)<pow(PHI,2)) and (abs(y/x)<pow(PHI,2)): # corner angles
        if   (x>0) and (y>0):
            return(1)
        elif (x<0) and (y>0):
            return(3)
        elif (x<0) and (y<0):
            return(5)
        elif (x>0) and (y<0):
            return(7)
    else: # square angles
        if   (x>0) and (abs(x)>abs(y)):
            return(int(0))
        elif (y>0) and (abs(y)>abs(x)):
            return(2)
        elif (x<0) and (abs(x)>abs(y)):
            return(4)
        elif (y<0) and (abs(y)>abs(x)):
            return(6)
        
RESIZE_FACTOR=2
SLIC_SPACE= 3
SLIC_SPACE= SLIC_SPACE*RESIZE_FACTOR

THREVAL= 60
RASMVAL= 160

CHANNEL= 2

def draw(img): # draw the bitmap
    plt.figure(dpi=600)
    plt.grid(False)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        
filename= sys.argv[1]
#filename= 'topanribut.png'
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(image, (RESIZE_FACTOR*image.shape[1], RESIZE_FACTOR*image.shape[0]), interpolation=cv.INTER_LINEAR)
image= resz.copy()
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU)
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
#_, gray= cv.threshold(selective_eroded, 0, THREVAL, cv.THRESH_TRIANGLE) # works better with dynamic-selective erosion
#draw(gray)
render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# Tampilkan grayscale
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')


_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernel = np.ones((2,2), np.uint8)

# 1) Opening → erosi, lalu dilasi → bantu buka lubang
opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

# 2) Closing → dilasi, lalu eroai → jaga keutuhan huruf
closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=1)

# Visualisasi
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Thresholded (Otsu)")
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Opened (2x2 kernel)")
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Opened + Closed")
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


#SLIC
cue = gray.copy()
slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = SLIC_SPACE)
slic.iterate()
mask= slic.getLabelContourMask()
result_mask = cv.bitwise_and(cue, mask)
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

# moments calculation for each superpixels, either voids or filled (in-stroke)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
moments_void = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue[j,i]!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render[j,i,0]= 140-(10*(lbls[j,i]%6))
        else:
            moments_void[lbls[j,i]] = np.append(moments_void[lbls[j,i]], np.array([[i,j]]), axis=0)

#moments[0][1] = [0,0] # random irregularities, not quite sure why
# some badly needed 'sanity' check
def remove_zeros(moments):
    temp=[]
    v= len(moments)
    if v==1:
        return temp
    else:
        for p in range(v):
            if moments[p][0]!=0. and moments[p][1]!=0.:
                temp.append(moments[p])
        return temp

for n in range(len(moments)):
    moments[n]= remove_zeros(moments[n])

# draw(render)

######## // image preprocessing ends here

# generating nodes
scribe= nx.Graph() # start anew, just in case

# valid superpixel
filled=0
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE ): # remove spurious superpixel with area less than 2 px 
        cx= int( np.mean( [array[0] for array in moments[n]] )) # centroid
        cy= int( np.mean( [array[1] for array in moments[n]] ))
        if (cue[cy,cx]!=0):
            render[cy,cx,1] = 255 
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), hurf='', pos_bitmap=(cx,cy), pos_render=(cx,-cy), color='#FFA500', rasm=True)
            #print(f'point{n} at ({cx},{cy})')
            filled=filled+1

def pdistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# connected componentscv.circle(disp, pos[compodef line_iterator(img, point0, point1):
from dataclasses import dataclass, field
from typing import List
from typing import Optional

@dataclass
class ConnectedComponents:
    rect: (int,int,int,int) # from bounding rectangle
    centroid: (int,int) # centroid moment
    area: Optional[int] = field(default=0)
    nodes: List[int] = field(default_factory=list)
    mat: Optional[np.ndarray] = field(default=None, repr=False)
    node_start: Optional[int] = field(default=-1)    # right-up
    distance_start: Optional[int] = field(default=0) # right-up
    node_end: Optional[int] = field(default=-1)      # left-down
    distance_end: Optional[int] = field(default=0)   # left-down


pos = nx.get_node_attributes(scribe,'pos_bitmap')
components=[]
for n in range(scribe.number_of_nodes()):
    # fill
    seed= pos[n]
    ccv= gray.copy()
    cv.floodFill(ccv, None, seed, RASMVAL, loDiff=(5), upDiff=(5))
    _, ccv = cv.threshold(ccv, 100, RASMVAL, cv.THRESH_BINARY)
    mu= cv.moments(ccv)
    if mu['m00'] > pow(SLIC_SPACE,2)*PHI:
        mc= (int(mu['m10'] / (mu['m00'])), int(mu['m01'] / (mu['m00'])))
        area = mu ['m00']
        pd= pdistance(seed, mc)
        node_start = n
        box= cv.boundingRect(ccv)
        # append keypoint if the component already exists
        found=0
        for i in range(len(components)):
            if components[i].centroid==mc:
                components[i].nodes.append(n)
                # calculate the distance
                tvane= freeman(seed[0]-mc[0], mc[1]-seed[1] )
                #if seed[0]>mc[0] and pd>components[i].distance_start and (tvane==2 or tvane==4): # potential node_start for long rasm
                if seed[0]>mc[0] and pd>components[i].distance_start: # potential node_start
                    components[i].distance_start= pd
                    components[i].node_start= n
                elif seed[0]<mc[0] and pd>components[i].distance_end: # potential node_end
                    components[i].distance_end = pd
                    components[i].node_end= n
                found=1
                # print(f'old node[{n}] with component[{i}] at {mc} from {components[i].centroid} distance: {pd})')
                break
        if (found==0):
            components.append(ConnectedComponents(box, mc))
            idx= len(components)-1
            components[idx].nodes.append(n)
            components[idx].mat = ccv.copy()
            components[idx].area = int(mu['m00']/THREVAL)
            if seed[0]>mc[0]:
                components[idx].node_start= n
                components[idx].distance_start= pd
            else:
                components[idx].node_end= n
                components[idx].distance_end= pd
            #print(f'new node[{n}] with component[{idx}] at {mc} from {components[idx].centroid} distance: {pd})')


components = sorted(components, key=lambda x: x.centroid[0], reverse=True)
# for n in len(components):
#     for i in components[n].nodes:
#         distance= pdistance(components[n].centroid, pos[i])
#         print(f'{i}: {distance}')

# # drawing the starting node (bitmap level)
# disp = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# for n in range(len(components)):
#     #print(f'{n} at {components[n].centroid} size {components[n].area}')
#     # draw green line for rasm at edges, color the rasm brighter
#     if components[n].area>4*PHI*pow(SLIC_SPACE,2):
#         disp= cv.bitwise_or(disp, cv.cvtColor(components[n].mat,cv.COLOR_GRAY2BGR))
#         seed= components[n].centroid
#         cv.circle(disp, seed, 2, (0,0,120), -1)
#         if components[n].node_start!=-1:
#             cv.circle(disp, pos[components[n].node_start], 2, (0,120,0), -1)
#         if components[n].node_end!=-1:
#             cv.circle(disp, pos[components[n].node_end], 2, (120,0,0), -1)
#         r= components[n].rect[0]+int(components[n].rect[2])
#         l= components[n].rect[0]
#         if l<width and r<width: # did we ever went beyond the frame?
#             for j1 in range(int(SLIC_SPACE*PHI),height-int(SLIC_SPACE*PHI)):
#                 disp[j1,r,1]= 120
#             for j1 in range(int(SLIC_SPACE*pow(PHI,3)),height-int(SLIC_SPACE*pow(PHI,3))):
#                 disp[j1,l,1]= 120
#     else:        
#         m= components[n].centroid[1]
#         i= components[n].centroid[0]
#         # draw blue line for shakil 'connection'
#         for j2 in range(int(m-(2*SLIC_SPACE*PHI)), int(m+(2*SLIC_SPACE*PHI))):
#             if j2<height and j2>0: 
#                 disp[j2,i,1]= RASMVAL/2
# draw(disp) 



# SKELETON
skeleton_components = []

for n in range(len(components)):
    binary_mat = (components[n].mat == RASMVAL).astype(np.uint8)
    skeleton = skeletonize(binary_mat)
    skeleton_components.append(skeleton)

# Gabungkan semua skeleton menjadi satu gambar
if len(skeleton_components) > 0:
    combined_skeleton = np.zeros_like(skeleton_components[0], dtype=np.uint8)

    for skeleton in skeleton_components:
        combined_skeleton |= skeleton
else:
    combined_skeleton = None

# Periksa apakah skeleton kosong sebelum menampilkan
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_skeleton, cmap='gray')
    plt.title("Skeletonization")
    plt.axis('off')
    plt.show()
else:
    print("Skeleton kosong, tidak ada data untuk ditampilkan.")
    

# DETEKSI TEPI (EDGE DETECTION)
edges = cv.Canny(gray, 30, 30)

# Temukan kontur berdasarkan gambar hasil deteksi tepi
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Opsional: tampilkan hasil deteksi tepi untuk verifikasi
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.show()

# def evaluasi_skeleton(skeleton, edge):
#     panjang = np.sum(skeleton > 0)
#     num_labels, _ = cv.connectedComponents(skeleton.astype(np.uint8))
#     overlap = np.logical_and(skeleton, edge).sum()
#     return {
#         'panjang_skeleton': panjang,
#         # 'jumlah_komponen': num_labels - 1,
#         'overlap_dengan_edge': overlap
#     }

# hasil = evaluasi_skeleton(combined_skeleton, edges)
# for k, v in hasil.items():
#     print(f"{k}: {v}")






# TRAVELLING SALESMAN PROBLEM (TSP)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
from collections import defaultdict
from skimage.morphology import skeletonize

# ====================
# STEP 1: Dapatkan combined_skeleton dari komponen (BUKAN dari thinning OpenCV)
# ====================

skeleton_components = []

for n in range(len(components)):
    binary_mat = (components[n].mat == RASMVAL).astype(np.uint8)
    skeleton = skeletonize(binary_mat).astype(np.uint8)
    skeleton_components.append(skeleton)

if len(skeleton_components) > 0:
    combined_skeleton = np.zeros_like(skeleton_components[0], dtype=np.uint8)
    for skeleton in skeleton_components:
        combined_skeleton |= skeleton
else:
    combined_skeleton = None

# ====================
# Fungsi bantu: buat graf dari tetangga 8-konektivitas skeleton
# ====================
def build_skeleton_graph(skeleton_img):
    h, w = skeleton_img.shape
    G = nx.Graph()
    coords = np.column_stack(np.where(skeleton_img > 0))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skeleton_img[ny, nx_] > 0:
                    p1 = (x, y)
                    p2 = (nx_, ny)
                    dist = euclidean(p1, p2)
                    G.add_edge(p1, p2, weight=dist)
    return G

# ====================
# STEP 2: Jalankan TSP jika skeleton valid
# ====================
JARAK_MAKSIMUM = 4
LOOP_RADIUS = 6

if combined_skeleton is not None and np.any(combined_skeleton > 0):
    points = np.column_stack(np.where(combined_skeleton > 0))[:, ::-1]
    points = [tuple(p) for p in points]

    if len(points) > 1:
        G = build_skeleton_graph(combined_skeleton)

        # Deteksi loop
        loop_nodes = set()
        for cycle in nx.cycle_basis(G):
            total_len = 0
            for i in range(len(cycle)):
                a = cycle[i]
                b = cycle[(i + 1) % len(cycle)]
                if G.has_edge(a, b):
                    total_len += G[a][b]['weight']
            # Tambahkan batasan jumlah titik dan panjang siklus
            if 5 <= len(cycle) <= 80 and total_len <= LOOP_RADIUS * len(cycle) * 1.5:
                loop_nodes.update(cycle)

        # Konversi ke indeks untuk digunakan di algoritma
        loop_nodes_idx = [i for i, p in enumerate(points) if p in loop_nodes]

        def tsp_greedy_with_revisit(points, loop_nodes_idx, max_visits=2):
            if not points:
                return []

            visits = [0] * len(points)
            tsp_path = []

            start_idx = np.argmax([p[1] for p in points])  # Paling kanan
            current_idx = start_idx
            tsp_path.append(points[current_idx])
            visits[current_idx] += 1

            for _ in range(len(points) * max_visits):
                current_point = points[current_idx]
                min_dist = float('inf')
                next_idx = None


# MEMPRIORITASKAN TITIK YANG PALING DEKAT
                for i, p in enumerate(points):
                    limit = max_visits if i in loop_nodes_idx else 1
                    if visits[i] >= limit:
                        continue
                    dist = euclidean(current_point, p)
                    if dist < min_dist:
                        min_dist = dist
                        next_idx = i
                  
                        
# JIKA PERCABANGAN MEMPRIORITASKAN YANG HORIZONTAL DIBANDING VERTIKAL
                # for i, p in enumerate(points):
                #     limit = max_visits if i in loop_nodes_idx else 1
                #     if visits[i] >= limit:
                #         continue
                
                #     dx = abs(current_point[0] - p[0])  # perbedaan X (horizontal)
                #     dy = abs(current_point[1] - p[1])  # perbedaan Y (vertikal)
                #     dist = euclidean(current_point, p)
                
                #     # Tambahkan penalti kecil jika gerakannya lebih vertikal
                #     direction_penalty = 0.1 * (dy > dx)  # 0.1 penalti jika gerak lebih vertikal
                #     effective_dist = dist + direction_penalty
                
                #     if effective_dist < min_dist:
                #         min_dist = effective_dist
                #         next_idx = i

                if next_idx is None:
                    break

                visits[next_idx] += 1
                tsp_path.append(points[next_idx])
                current_idx = next_idx

            return tsp_path

        tsp_full = tsp_greedy_with_revisit(points, loop_nodes_idx, max_visits=2)

        # Pisah sub-path berdasarkan jarak
        sub_paths = []
        current_subpath = [tsp_full[0]]
        for i in range(1, len(tsp_full)):
            dist = euclidean(tsp_full[i - 1], tsp_full[i])
            if dist > JARAK_MAKSIMUM:
                if len(current_subpath) > 1:
                    sub_paths.append(current_subpath)
                current_subpath = [tsp_full[i]]
            else:
                current_subpath.append(tsp_full[i])
        if len(current_subpath) > 1:
            sub_paths.append(current_subpath)

        # Hitung jumlah kunjungan
        visit_counts = defaultdict(int)
        for pt in tsp_full:
            visit_counts[pt] += 1

        # BALIK sub-path: tukar titik awal <-> akhir
            sub_paths = [list(reversed(p)) for p in sub_paths]


        # ====================
        # STEP 3: VISUALISASI
        # ====================
        skeleton_rgb = cv.cvtColor((combined_skeleton * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton_rgb)
        total_distance = 0
        colors = ['r', 'y', 'g', 'b', 'm']

        for idx, path in enumerate(sub_paths):
            is_loop = any(pt in loop_nodes for pt in path)
            color = 'g' if is_loop else 'r'

            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                dist = euclidean(p1, p2)
                total_distance += dist

                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color + '-', linewidth=1)
                visit_color = 'purple' if visit_counts[p1] > 1 else 'red'
                # Plot titik dan tuliskan jumlah kunjungan
                plt.plot(p1[0], p1[1], 'o', color=visit_color, markersize=3)
                plt.text(p1[0]+1, p1[1], f"{visit_counts[p1]}", color='white', fontsize=6)


            plt.plot(path[0][0], path[0][1], 'bo', markersize=5)  # Start
            plt.plot(path[-1][0], path[-1][1], 'yo', markersize=5)  # End

        plt.title(f"TSP with Revisit (Loop Aware)\nTotal Distance: {total_distance:.2f}")
        plt.axis('off')
        plt.show()

        # Tampilkan titik yang dikunjungi lebih dari sekali
        revisited_points = [pt for pt, count in visit_counts.items() if count > 1]
        print(f"\nTotal titik dikunjungi >1x: {len(revisited_points)}")
        for i, pt in enumerate(revisited_points[:10]):
            print(f"{i+1}. Titik {pt} dikunjungi {visit_counts[pt]} kali")
        else:
            print("Skeleton kosong, tidak ada data untuk diproses.")
            
            
        # Tampilkan catatan titik yang dikunjungi
        visited = {
            'visited_once': [],  # titik-titik yang hanya dilewati sekali.
            'visited_twice': [], # titik yang dilewati dua kali (looping).
            'visited_more': []   # titik yang dilewati lebih dari 2x (bisa jadi karena error)
        }

        for pt, count in visit_counts.items():
            if count == 1:
                visited['visited_once'].append(pt)
            elif count == 2:
                visited['visited_twice'].append(pt)
            elif count > 2:
                visited['visited_more'].append((pt, count))
    
    
# ========================
# SIMPAN SUB-PATH
# ========================
print("\nSub-path details:")

global_counter = 1
for idx, path in enumerate(sub_paths):
    # Cek apakah sub-path ini mengandung titik dari loop
    is_loop = any(pt in loop_nodes for pt in path)
    
    # Hitung panjang total
    total_length = 0
    for i in range(1, len(path)):
        total_length += euclidean(path[i], path[i - 1])
    
    print(f"\nSub-path {idx+1}: (loop: {is_loop}, panjang: {total_length:.2f})")
    for i in range(len(path)):
        pt = tuple(int(v) for v in path[i])
        if i == 0:
            dist = 0.00
        else:
            dist = euclidean(path[i], path[i - 1])
        print(f"{global_counter}. Point {pt} | Jarak: {dist:.2f}")
        global_counter += 1










# FREEMAN CHAIN CODE
def direction_code(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    directions = {
        (1, 0): 0, # kanan
        (1, -1): 1, #kanan atas
        (0, -1): 2, #atas
        (-1, -1): 3, #kiri atas
        (-1, 0): 4, #kiri
        (-1, 1): 5, #kiri bawah
        (0, 1): 6, #bawah
        (1, 1): 7 #kanan bawah
    }
    return directions.get((dx, dy), -1)  # -1 jika bukan tetangga langsung

# Proses Freeman Chain Code untuk setiap subpath
print("\nFreeman Chain Code untuk setiap sub-path:")

for idx, path in enumerate(sub_paths):
    chain_code = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        code = direction_code(p1, p2)
        
        if code != -1:
            chain_code.append(code)
        else:
            # Jika bukan tetangga langsung, lakukan pemecahan langkah dengan batas langkah
            x1, y1 = p1
            x2, y2 = p2
            steps = 0
            max_steps = 1000  # untuk menghindari infinite loop
            while (x1, y1) != (x2, y2) and steps < max_steps:
                dx = np.sign(x2 - x1)
                dy = np.sign(y2 - y1)
                next_x = x1 + dx
                next_y = y1 + dy
                code = direction_code((x1, y1), (next_x, next_y))
                if code != -1:
                    chain_code.append(code)
                    x1, y1 = next_x, next_y
                else:
                    print(f"Gagal menentukan arah dari ({x1}, {y1}) ke ({next_x}, {next_y})")
                    break
                steps += 1
            if steps >= max_steps:
                print(f"⚠️  Langkah melebihi batas pada sub-path {idx + 1}, antara titik {p1} dan {p2}")
    
    print(f"Sub-path {idx + 1}:", chain_code)
    # print("Chain Code:", chain_code)





# PEMOTONGAN HURUF
def tsp_path(points):
    """Greedy TSP path, from leftmost to rightmost."""
    if len(points) == 0:
        return []

    visited = [False] * len(points)
    path = []

    # Mulai dari titik paling kiri
    start_idx = np.argmin([p[0] for p in points])
    current = start_idx
    path.append(points[current])
    visited[current] = True

    for _ in range(len(points) - 1):
        current_point = points[current]
        min_dist = float('inf')
        next_point = -1
        for idx, point in enumerate(points):
            if not visited[idx]:
                dist = np.linalg.norm(np.array(current_point) - np.array(point))
                if dist < min_dist:
                    min_dist = dist
                    next_point = idx
        if next_point == -1:
            break
        visited[next_point] = True
        path.append(points[next_point])
        current = next_point

    return path

def split_tsp_path_by_distance(path, max_distance=JARAK_MAKSIMUM):
    """Pisahkan path menjadi subpath jika jarak antar-node melebihi max_distance."""
    if len(path) < 2:
        return [path]

    subpaths = []
    current_subpath = [path[0]]

    for i in range(1, len(path)):
        prev_point = path[i - 1]
        curr_point = path[i]
        distance = np.linalg.norm(np.array(prev_point) - np.array(curr_point))
        if distance > max_distance:
            subpaths.append(current_subpath)
            current_subpath = [curr_point]
        else:
            current_subpath.append(curr_point)

    if current_subpath:
        subpaths.append(current_subpath)

    return subpaths

# Ambil titik skeleton
skeleton_points = np.argwhere(combined_skeleton > 0)
skeleton_points = [(x[1], x[0]) for x in skeleton_points]  # (x, y)

# Buat jalur TSP
tsp_path_full = tsp_path(skeleton_points)

# Pisahkan jalur TSP berdasarkan threshold jarak
subpaths = split_tsp_path_by_distance(tsp_path_full, max_distance=JARAK_MAKSIMUM)

# Buat folder output jika belum ada
output_folder = 'output_potongan_huruf'
os.makedirs(output_folder, exist_ok=True)

huruf_terpotong = []  # Menyimpan semua hasil potongan
for idx, path in enumerate(sub_paths):
    # Ambil bounding box dari titik-titik subpath
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    min_x = max(min(xs) - 2, 0)
    max_x = min(max(xs) + 3, combined_skeleton.shape[1])
    min_y = max(min(ys) - 2, 0)
    max_y = min(max(ys) + 3, combined_skeleton.shape[0])

    # Potong area dari skeleton asli
    potongan = combined_skeleton[min_y:max_y, min_x:max_x]
    huruf_terpotong.append(potongan)

    # Simpan sebagai gambar
    filename = os.path.join(output_folder, f"huruf_{idx+1}.png")
    cv.imwrite(filename, (potongan * 255).astype(np.uint8))
    print(f"Subpath {idx+1} disimpan: {filename}")

# Visualisasi
plt.figure(figsize=(4 * len(huruf_terpotong), 6))
for i, img in enumerate(huruf_terpotong):
    plt.subplot(1, len(huruf_terpotong), i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Potongan {i+1}")

plt.suptitle(f"Pemotongan Huruf Berdasarkan Path TSP\nTotal Subpath: {len(huruf_terpotong)}", fontsize=14)
plt.tight_layout()
plt.show()
