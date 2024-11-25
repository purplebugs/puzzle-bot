# Import necessary libraries
import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import splprep, splev


def get_line_angle(x1, y1, x2, y2):
    # Calculate the angle of the line in degrees (0-180)
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    # if angle < 0:
    #     angle += 180
    return angle
    
def get_line_length(x1, y1, x2, y2):
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length

    
def find_four_corners(corners, component, image_with_contours):
#return the four corners
    # Calculate the angle and length of each line
    lines = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            a = corners[i]
            b = corners[j]
            angle = get_line_angle(a[0], a[1], b[0], b[1])
            length = get_line_length(a[0], a[1], b[0], b[1])
            if length < 0.1:
                continue
            lines.append({'p1': a, 'p2': b, 'length': length, 'angle': angle})

    print(f"Number of lines: {len(lines)}")
    for line in lines:
        print(f"Line: {line['p1']} -> {line['p2']} | Length: {line['length']} | Angle: {line['angle']}")

    lines = sorted(lines, key=lambda x: x['length'], reverse=True)
    
    min_length = 0.6 * min(component['width'], component['height'])
    filtered_lines = []
    for line in lines:
        if line['length'] >= min_length:
            filtered_lines.append(line)
        else:
            print(f"Line removed: {line['p1']} -> {line['p2']} | Length: {line['length']} | Angle: {line['angle']}")
    
    lines = filtered_lines

    tmp_img = image_with_contours.copy()
    for line in lines:
        cv2.line(tmp_img, line['p1'], line['p2'], (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Lines')
    plt.axis('off')
    plt.show()

    # investigating line pairs (candidates)
    candidates = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a = lines[i]
            b = lines[j]
            length = 0 #a['length'] + b['length'] 
            for p1 in [a['p1'], a['p2']]:
                for p2 in [b['p1'], b['p2']]:
                    length += get_line_length(p1[0], p1[1], p2[0], p2[1])
            a_angle = a['angle'] % 180
            b_angle = b['angle'] % 180
            min_delta_angle = min(abs(a_angle - b_angle), abs(a_angle - (b_angle - 180)), abs(a_angle - (b_angle + 180)))
            if min_delta_angle < 7.5: # and minimum_length > 0.5 * min(component['width'], component['height']):
                candidates.append({'l1': a, 'l2': b, 'delta_angle': min_delta_angle, 'length': length})

    candidates = sorted(candidates, key=lambda x: x['length'], reverse=True)

    tmp_img = image_with_contours.copy()
    for candidate in candidates:
        cv2.line(tmp_img, candidate['l1']['p1'], candidate['l1']['p2'], (255, 0, 0), 4)
        cv2.line(tmp_img, candidate['l2']['p1'], candidate['l2']['p2'], (255, 0, 0), 4)
        print(f"Candidate: {candidate['l1']['p1']} -> {candidate['l1']['p2']} | {candidate['l2']['p1']} -> {candidate['l2']['p2']} | Delta Angle: {candidate['delta_angle']:.1f} | Length: {candidate['length']:.1f} | Angles: {candidate['l1']['angle']:.1f} | {candidate['l2']['angle']:.1f}")
    plt.imshow(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
    plt.title('Line Pairs (candidates)')
    plt.axis('off')
    plt.show()


    print(f"Number of candidate lines: {len(candidates)}")
    for candidate in candidates:
        print(f"Candidate: {candidate['l1']['p1']} -> {candidate['l1']['p2']} | {candidate['l2']['p1']} -> {candidate['l2']['p2']} | Delta Angle: {candidate['delta_angle']:.1f} | Length: {candidate['length']:.1f}")

    #investigating pairs of line-pairs to form a square idetified by 4 corners
    candidate_pairs = []
    for i in range(len(candidates)):
        candidate = candidates[i]
        print(f"* Candidate: {candidate['l1']['p1']} -> {candidate['l1']['p2']} | {candidate['l2']['p1']} -> {candidate['l2']['p2']} | Delta Angle: {candidate['delta_angle']:.1f}")
        points_A = [tuple(map(float, p)) for p in [candidate['l1']['p1'], candidate['l1']['p2'], candidate['l2']['p1'], candidate['l2']['p2']]]
        for j in range(i + 1, len(candidates)):
            other_candidate = candidates[j]
            points_B = [tuple(map(float, p)) for p in [other_candidate['l1']['p1'], other_candidate['l1']['p2'], other_candidate['l2']['p1'], other_candidate['l2']['p2']]]
            if set(points_A) == set(points_B):
                angle_A1 = get_line_angle(candidate['l1']['p1'][0], candidate['l1']['p1'][1], candidate['l1']['p2'][0], candidate['l1']['p2'][1])
                angle_A2 = get_line_angle(candidate['l2']['p1'][0], candidate['l2']['p1'][1], candidate['l2']['p2'][0], candidate['l2']['p2'][1])
                angle_B1 = get_line_angle(other_candidate['l1']['p1'][0], other_candidate['l1']['p1'][1], other_candidate['l1']['p2'][0], other_candidate['l1']['p2'][1])
                angle_B2 = get_line_angle(other_candidate['l2']['p1'][0], other_candidate['l2']['p1'][1], other_candidate['l2']['p2'][0], other_candidate['l2']['p2'][1])
                # if (angle_A1 - angle_B1 + 180) % 360 > 70 and (angle_A1 - angle_B1 + 180) % 360 < 110 and \
                #    (angle_A2 - angle_B2 + 180) % 360 > 70 and (angle_A2 - angle_B2 + 180) % 360 < 110:
                angle_diff = abs(angle_A1 - angle_B1) % 180
                print(f"** matching candidate: {candidate['l1']['p1']} -> {candidate['l1']['p2']} @ {angle_A1:.1f} | {other_candidate['l1']['p1']} -> {other_candidate['l1']['p2']} @ {angle_B1:.1f} | Angle Diff: {angle_diff:.1f}")
                # print(f"** Matching candidate: {candidate['l1']['p1']} -> {candidate['l1']['p2']} | {candidate['l2']['p1']} -> {candidate['l2']['p2']} | Delta Angle: {candidate['delta_angle']:.1f}")
                if abs(angle_diff-90) < 7.5:
                    candidate_pairs.append({'c1': candidate, 'c2': other_candidate, 'length': candidate['length'] + other_candidate['length']})
                    
    print(f"Number of candidate pairs: {len(candidate_pairs)}")
    if len(candidate_pairs) == 0:
        return None
    new_candidate_pairs = []
    for candidate_pair in candidate_pairs:
        length_diff = abs((candidate_pair['c1']['l1']['length']+candidate_pair['c1']['l2']['length'])/2 - (candidate_pair['c2']['l1']['length']+candidate_pair['c2']['l2']['length'])/2)
        delta_angle = abs(candidate_pair['c1']['delta_angle'] + candidate_pair['c2']['delta_angle'])/2
        corners = [candidate_pair['c1']['l1']['p1'], candidate_pair['c1']['l1']['p2'], candidate_pair['c1']['l2']['p1'], candidate_pair['c1']['l2']['p2']]
        # Sort the four corners and return them sorted
        corners.sort(key=lambda x: x[0])  # Sort by x-coordinate
        top_two = sorted(corners[:2], key=lambda x: x[1])  # Sort top two points by y-coordinate
        bottom_two = sorted(corners[2:], key=lambda x: x[1], reverse=True)  # Sort bottom two points by y-coordinate
        corners = top_two + bottom_two
        
        total_length = candidate_pair['c1']['l1']['length'] + candidate_pair['c1']['l2']['length'] + candidate_pair['c2']['l1']['length'] + candidate_pair['c2']['l2']['length']
        # total_length2 = candidate_pair['c1']['length'] + candidate_pair['c2']['length'] # old value, noe longer needed ?
        relative_length_diff = (length_diff*100.0)/total_length
        print(f"Candidate pair: {candidate_pair['c1']['l1']['p1']} -> {candidate_pair['c1']['l1']['p2']} | {candidate_pair['c2']['l1']['p1']} -> {candidate_pair['c2']['l1']['p2']} => Length: {total_length:.1f} | Length Diff: {length_diff:.1f}/{relative_length_diff:.1f} | Delta Angle: {delta_angle:.1f}")
        candidate_pair['c1']['length'] = total_length
        candidate_pair['c2']['length'] = total_length
        if relative_length_diff > 5:
            print(f"Length diff too high: {relative_length_diff:.1f}% => removing candidate pair")
        else:
            new_candidate_pairs.append(candidate_pair)
    candidate_pairs = new_candidate_pairs


    # sort candidate pairs by most similar line length on all for sides
    # candidate_pairs.sort(key=lambda x:abs((x['c1']['l1']['length']+x['c1']['l2']['length'])/2 - (x['c2']['l1']['length']+x['c2']['l2']['length'])/2))

    # sort candidate pairs by most similar line length on all for sides
    candidate_pairs.sort(key=lambda x:abs(x['c1']['length'] + x['c2']['length'])/2, reverse=True)
    print("-------------------------------------")
   
    candidate_pair = candidate_pairs[0]
    # for candidate_pair in candidate_pairs:
    tmp_img = image_with_contours.copy()
    cv2.line(tmp_img, candidate_pair['c1']['l1']['p1'], candidate_pair['c1']['l1']['p2'], (0, 0, 255), 3)
    cv2.line(tmp_img, candidate_pair['c1']['l2']['p1'], candidate_pair['c1']['l2']['p2'], (0, 0, 255), 3)
    cv2.line(tmp_img, candidate_pair['c2']['l1']['p1'], candidate_pair['c2']['l1']['p2'], (0, 0, 255), 3)
    cv2.line(tmp_img, candidate_pair['c2']['l2']['p1'], candidate_pair['c2']['l2']['p2'], (0, 0, 255), 3)
    print(f"Candidate pair: {candidate_pair['c1']['l1']['p1']} -> {candidate_pair['c1']['l1']['p2']} | {candidate_pair['c2']['l1']['p1']} -> {candidate_pair['c2']['l1']['p2']} | Length: {candidate_pair['c1']['length']:.1f} | {candidate_pair['c2']['length']:.1f} | Angles: {candidate_pair['c1']['l1']['angle']:.1f} | {candidate_pair['c1']['l2']['angle']:.1f} | {candidate_pair['c2']['l1']['angle']:.1f} | {candidate_pair['c2']['l2']['angle']:.1f} | Delta Angles: {candidate_pair['c1']['delta_angle']:.1f} | {candidate_pair['c2']['delta_angle']:.1f}")
    print(f"    => Lengths: {candidate_pair['c1']['l1']['p1']} -> {candidate_pair['c1']['l1']['p2']} | {candidate_pair['c2']['l1']['p1']} -> {candidate_pair['c2']['l1']['p2']} | Lengths: {candidate_pair['c1']['l1']['length']:.1f} | {candidate_pair['c1']['l2']['length']:.1f} | {candidate_pair['c2']['l1']['length']:.1f} | {candidate_pair['c2']['l2']['length']:.1f}")
    
    total_length = sum([
        get_line_length(candidate_pair['c1']['l1']['p1'][0], candidate_pair['c1']['l1']['p1'][1], candidate_pair['c1']['l1']['p2'][0], candidate_pair['c1']['l1']['p2'][1]),
        get_line_length(candidate_pair['c1']['l2']['p1'][0], candidate_pair['c1']['l2']['p1'][1], candidate_pair['c1']['l2']['p2'][0], candidate_pair['c1']['l2']['p2'][1]),
        get_line_length(candidate_pair['c2']['l1']['p1'][0], candidate_pair['c2']['l1']['p1'][1], candidate_pair['c2']['l1']['p2'][0], candidate_pair['c2']['l1']['p2'][1]),
        get_line_length(candidate_pair['c2']['l2']['p1'][0], candidate_pair['c2']['l2']['p1'][1], candidate_pair['c2']['l2']['p2'][0], candidate_pair['c2']['l2']['p2'][1])
    ])
    print(f"Total length: {total_length:.2f}")
    plt.imshow(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
    plt.title('Candidate Pairs')
    plt.axis('off')
    plt.show()

    # Sort the four corners and return them sorted
    corners = [candidate_pairs[0]['c1']['l1']['p1'], candidate_pairs[0]['c1']['l1']['p2'], candidate_pairs[0]['c1']['l2']['p1'], candidate_pairs[0]['c1']['l2']['p2']]
    # Sort the four corners and return them sorted
    corners.sort(key=lambda x: x[0])  # Sort by x-coordinate
    top_two = sorted(corners[:2], key=lambda x: x[1])  # Sort top two points by y-coordinate
    bottom_two = sorted(corners[2:], key=lambda x: x[1], reverse=True)  # Sort bottom two points by y-coordinate
    corners = top_two + bottom_two
    return corners


    # return candidate_pairs[0]['c1']['l1']['p1'], candidate_pairs[0]['c1']['l1']['p2'], candidate_pairs[0]['c1']['l2']['p1'], candidate_pairs[0]['c1']['l2']['p2']

def find_corners_by_approxPoly(component, image_with_contours, eps=0.01):
    # Approximate contours and find corners
    print("Approximating contours and finding corners...")
    corners_list = []
    contour = component['contour']
    box = component['box']
    width = component['width']
    height = component['height']
        # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    # Apply contour approximation
    epsilon = eps * perimeter  # Adjusted epsilon for better corner detection
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # Extract corners from the approximated contour
    corners = approx.reshape(-1, 2)
    # for corner in corners:
    #    matching_box_point = find_closes_corner(corner, component, search_radius)
    #    if matching_box_point is not None:
    #        corners_list.append(corner)

    corners_list = find_four_corners(corners, component, image_with_contours)
    if corners_list is None:
        return None

    # Optionally, draw the corners on the image
    for corner in corners_list:
        x, y = corner
        cv2.circle(image_with_contours, (x, y), 5, (0, 0, 255), -1)
    print(f"Number of corners: {len(corners_list)}")

    # Display the image with contours and corners
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contours with Corners')
    plt.axis('off')
    plt.show()

    return np.array(corners_list)
