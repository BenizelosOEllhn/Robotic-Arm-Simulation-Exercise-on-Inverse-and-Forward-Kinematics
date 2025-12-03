import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# ==========================================
# PART 1: MATH LIBRARY
# ==========================================

def hat(vec):
    v = vec.reshape((3,))
    return np.array([
        [0.,     -v[2],  v[1]],
        [v[2],   0.,    -v[0]],
        [-v[1],  v[0],   0.]
    ])

def exp_rotation(p):
    phi = p.reshape((3, 1))
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3, 3)
    a = phi / theta
    return (np.eye(3) * np.cos(theta)
            + (1. - np.cos(theta)) * a @ a.T
            + np.sin(theta) * hat(a))

def log_rotation(R):
    theta = np.arccos(max(-1., min(1., (np.trace(R) - 1.) / 2.)))
    if theta < 1e-12:
        return np.zeros((3, 1))
    mat = R - R.T
    r = np.array([mat[2, 1], mat[0, 2], mat[1, 0]]).reshape((3, 1))
    return theta / (2. * np.sin(theta)) * r

def exp_pose(tau):
    theta = np.linalg.norm(tau[:3, :])
    R = np.eye(3)
    p = np.zeros((3, 1))
    if not np.isclose(theta, 0.):
        r = tau[:3, :] / theta
        rho = tau[3:, :] / theta
        rh = hat(r)
        R = exp_rotation(tau[:3, :])
        p = (np.eye(3) * theta
             + (1. - np.cos(theta)) * rh
             + (theta - np.sin(theta)) * (rh @ rh)) @ rho
    else:
        p = tau[3:, :]
    return np.block([[R, p],
                     [np.zeros((1, 3)), 1.]])

def log_pose(T):
    R = T[:3, :3]
    p = T[:3, 3:]
    rt = log_rotation(R)
    theta = np.linalg.norm(rt)
    if np.allclose(theta, 0.):
        return np.block([[np.zeros((3, 1))], [p]])
    rh = hat(rt / theta)
    Ginv = (1./theta * np.eye(3)
            - 0.5 * rh
            + (1./theta - 0.5 / np.tan(theta/2.)) * (rh @ rh))
    return np.block([[rt],
                     [Ginv @ p * theta]])

def Adj(R, p):
    Tadj = np.zeros((6, 6))
    Tadj[:3, :3] = R
    Tadj[3:, 3:] = R
    Tadj[3:, :3] = hat(p) @ R
    return Tadj

def homogeneous(R, p=np.zeros((3, 1))):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = p.reshape((3, 1))
    return T

# ==========================================
# PART 2: ROBOT KINEMATICS & IK SOLVER
# ==========================================

# Dimensions (Meters)
d1 = 0.1695
d3 = 0.1155
d5 = 0.12783
d7 = 0.06598
z_off = 0.075   # tool_link offset
x_off = 0.008

z_total = d1 + d3 + d5 + d7 + z_off

# Home Matrix M
M = np.array([
    [1, 0, 0, x_off],
    [0, 1, 0, 0.0],
    [0, 0, 1, z_total],
    [0, 0, 0, 1]
])

# Screw Axes
h_j2 = d1
h_j4 = d1 + d3
h_j6 = d1 + d3 + d5

S_list = np.array([
    [ 0.0,  0.0,  1.0,    0.0,      0.0, 0.0    ],  # J1
    [ 0.0,  1.0,  0.0,   -h_j2,     0.0, 0.0    ],  # J2
    [ 0.0,  0.0,  1.0,    0.0,      0.0, 0.0    ],  # J3
    [ 0.0, -1.0,  0.0,    h_j4,     0.0, 0.0    ],  # J4
    [ 0.0,  0.0,  1.0,    0.0,      0.0, 0.0    ],  # J5
    [ 0.0, -1.0,  0.0,    h_j6,     0.0, 0.0    ],  # J6
    [ 0.0,  0.0,  1.0,    0.0,      0.0, 0.0    ],  # J7
]).T

JOINT_LIMITS_DEG = [
    (-160, 160), (-70, 115), (-170, 170),
    (-113, 75), (-170, 170), (-115, 115), (-180, 180)
]

def FK_PoE(thetas):
    T = np.eye(4)
    for i in range(7):
        twist = S_list[:, i].reshape((6, 1)) * thetas[i]
        T = T @ exp_pose(twist)
    return T @ M

def JacobianSpace(thetas):
    Js = np.zeros((6, 7))
    T = np.eye(4)
    Js[:, 0] = S_list[:, 0]
    for i in range(1, 7):
        twist_prev = S_list[:, i-1].reshape((6, 1)) * thetas[i-1]
        T = T @ exp_pose(twist_prev)
        R = T[:3, :3]
        p = T[:3, 3:]
        AdT = Adj(R, p)
        Js[:, i] = AdT @ S_list[:, i]
    return Js

def normalize_angles(thetas):
    return (thetas + np.pi) % (2 * np.pi) - np.pi

def check_limits(thetas):
    deg = np.degrees(thetas)
    for i, (val, (low, high)) in enumerate(zip(deg, JOINT_LIMITS_DEG)):
        if not (low <= val <= high):
            return False
    return True

def NewtonRaphsonIK(T_des, theta_guess):
    theta = np.copy(theta_guess)
    for _ in range(120):
        T_curr = FK_PoE(theta)
        T_rel = T_des @ np.linalg.inv(T_curr)
        V_err = log_pose(T_rel).reshape((6,))
        if (np.linalg.norm(V_err[:3]) < 1e-3 and
            np.linalg.norm(V_err[3:]) < 1e-3):
            return normalize_angles(theta), True
        Js = JacobianSpace(theta)
        theta += np.linalg.pinv(Js) @ V_err
    return theta, False

def solve_with_restarts(T_target, attempts=80):
    smart_guess = np.radians([0, 30, 0, -90, 0, 30, 0])
    q, conv = NewtonRaphsonIK(T_target, smart_guess)
    if conv and check_limits(q):
        return q, True, 0
    for i in range(attempts):
        rand_guess = np.random.uniform(-1.5, 1.5, 7)
        q, conv = NewtonRaphsonIK(T_target, rand_guess)
        if conv and check_limits(q):
            return q, True, i + 1
    return np.zeros(7), False, -1

# ==========================================
# PART 3: ROS 2 VISUALIZATION NODE
# ==========================================

class HW2Node(Node):
    def __init__(self):
        super().__init__('hw2_minimal_grasp_place')
        self.pub_joints = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        print("\n=== CALCULATING INVERSE KINEMATICS ===")
        np.set_printoptions(precision=4, suppress=True)

        # --------- Home (all zeros) ----------
        self.q_home = np.zeros(7)

        # --------- Grasp pose (Z=0.030) ----------
        self.T_grasp = np.array([
            [1,  0,  0, 0.175],
            [0, -1,  0, 0.025],
            [0,  0, -1, 0.030],
            [0,  0,  0, 1.000]
        ])
        self.q_grasp, ok_g, att_g = solve_with_restarts(self.T_grasp)
        if ok_g:
            print(f"\n[IK] Grasp Pose (Attempt {att_g}): SUCCESS")
            print("Joints (deg):", np.degrees(self.q_grasp))
        else:
            print("\n[IK] Grasp Pose: FAILED")

        # --------- Place pose (Rotated) ----------
        self.T_place = np.array([
            [ 0.707,  0.707,  0.000,  0.100], 
            [ 0.000,  0.000,  1.000,  0.080], 
            [ 0.707, -0.707,  0.000,  0.100], 
            [ 0.000,  0.000,  0.000,  1.000]
        ])

        self.q_place, ok_p, att_p = solve_with_restarts(self.T_place)
        if ok_p:
            print(f"\n[IK] Place Pose (Attempt {att_p}): SUCCESS")
            print("Joints (deg):", np.degrees(self.q_place))
        else:
            print("\n[IK] Place Pose: FAILED")

        # --------- Cube Logic ----------
        # Final pose definition
        p_cube_final = np.array([0.10, 0.125, 0.10])
        
        # Offsets are computed dynamically
        self.T_off = homogeneous(np.eye(3), np.zeros(3))

        self.cube_state = "table"
        self.cube_table_pos = np.array([0.175, 0.025, 0.025])
        self.cube_table_rot = np.eye(3)

        self.cube_final_pos = self.T_place[:3, 3]
        self.cube_final_rot = self.T_place[:3, :3]

        self.grasp_dist_thresh = 0.03
        self.place_dist_thresh = 0.005

        # Grasp point in tool frame
        self.T_grasp_point = homogeneous(np.eye(3), np.array([0, 0, -0.10]))

        # Animation Phases
        # 0: Home -> Grasp
        # 1: Grasp -> Place
        # 2: Hold
        self.phase = 0
        self.alpha = 0.0
        self.alpha_step = 0.005

        self.timer = self.create_timer(0.05, self.timer_callback)
        print("\n=== STARTING VISUALIZATION LOOP ===")

    # ---------- Helper functions ----------

    def get_quat(self, mat):
        r = R_scipy.from_matrix(mat)
        return r.as_quat()

    def create_marker(self, id, scale, color, pos, rot=np.eye(3), frame="base"):
        m = Marker()
        m.header.frame_id = frame
        m.id = id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.scale = Vector3(x=scale[0], y=scale[1], z=scale[2])
        m.color = ColorRGBA(
            r=float(color[0]),
            g=float(color[1]),
            b=float(color[2]),
            a=float(color[3])
        )
        m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        q = self.get_quat(rot)
        m.pose.orientation.x = float(q[0])
        m.pose.orientation.y = float(q[1])
        m.pose.orientation.z = float(q[2])
        m.pose.orientation.w = float(q[3])
        return m

    def interpolate(self, q_from, q_to, alpha):
        return (1.0 - alpha) * q_from + alpha * q_to

    # ---------- Main callback ----------

    def timer_callback(self):
        # 1. Select Motion Phase
        if self.phase == 0:
            q_from, q_to = self.q_home, self.q_grasp
        elif self.phase == 1:
            q_from, q_to = self.q_grasp, self.q_place
        else:
            q_from, q_to = self.q_place, self.q_place

        # 2. Interpolate
        current_q = self.interpolate(q_from, q_to, self.alpha)

        # 3. Publish Joints
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'joint1_to_base', 'joint2_to_joint1', 'joint3_to_joint2', 
            'joint4_to_joint3', 'joint5_to_joint4', 'joint6_to_joint5', 
            'joint7_to_joint6'
        ]
        msg.position = [float(x) for x in current_q]
        self.pub_joints.publish(msg)

        # 4. Compute TCP
        T_ee = FK_PoE(current_q)
        R_ee = T_ee[:3, :3]
        p_ee = T_ee[:3, 3]

        T_grasp_world = T_ee @ self.T_grasp_point
        p_grasp = T_grasp_world[:3, 3]

        # 5. Cube State Logic
        if self.cube_state == "table":
            cube_pos = self.cube_table_pos
            cube_rot = self.cube_table_rot

            dist = np.linalg.norm(p_grasp - cube_pos)
            if dist < self.grasp_dist_thresh:
                print(">>> Cube attached to gripper.")
                self.cube_state = "attached"
                T_cube_world = homogeneous(cube_rot, cube_pos)
                self.T_off = np.linalg.inv(T_ee) @ T_cube_world

        elif self.cube_state == "attached":
            T_cube = T_ee @ self.T_off
            cube_rot = T_cube[:3, :3]
            cube_pos = T_cube[:3, 3]

            if np.linalg.norm(cube_pos - self.cube_final_pos) < self.place_dist_thresh:
                print(">>> Cube placed.")
                self.cube_state = "placed"

        else: # "placed"
            cube_pos = self.cube_final_pos
            cube_rot = self.cube_final_rot

        # 6. Build Markers
        markers = MarkerArray()

        # [ID 1] Blue Cube (The object being moved)
        markers.markers.append(
            self.create_marker(
                id=1,
                scale=[0.05, 0.05, 0.05],
                color=[0.0, 0.0, 1.0, 1.0],
                pos=cube_pos,
                rot=cube_rot
            )
        )

        # [ID 2] Red Box (Container Walls)
        red_rot = np.array([
            [ 0,  1,  0],
            [-1,  0,  0],
            [ 0,  0,  1]
        ])
        markers.markers.append(
            self.create_marker(
                id=2,
                scale=[0.05, 0.20, 0.20],
                color=[0.8, 0.2, 0.2, 0.6],
                pos=[0.10, 0.125, 0.10],
                rot=red_rot
            )
        )

        # [ID 3] The Hole (Darker inner square)
        markers.markers.append(
            self.create_marker(
                id=3,
                scale=[0.05, 0.05, 0.05],
                color=[0.0, 0.0, 0.0, 0.8],
                pos=[0.10, 0.125, 0.10],
                rot=self.cube_final_rot
            )
        )

        # [ID 4] The Subtractor (Invisible masking cube)
        markers.markers.append(
            self.create_marker(
                id=4,
                scale=[0.05, 0.05, 0.05],
                color=[0.0, 0.0, 0.0, 0.0],
                pos=[0.10, 0.125, 0.10],
                rot=self.cube_final_rot
            )
        )

        self.pub_markers.publish(markers)

        # 7. Update Animation
        self.alpha += self.alpha_step

        if self.alpha >= 1.0:
            self.alpha = 0.0

            # Force state changes at end of phases if physics missed
            if self.phase == 0:
                if self.cube_state == "table":
                    print(">>> Forcing cube attach.")
                    self.cube_state = "attached"
                    T_cube_world = homogeneous(self.cube_table_rot, self.cube_table_pos)
                    self.T_off = np.linalg.inv(T_ee) @ T_cube_world

            elif self.phase == 1:
                if self.cube_state == "attached":
                    print(">>> Forcing cube placed.")
                    self.cube_state = "placed"

            if self.phase < 2:
                self.phase += 1


# ==========================================
# MAIN
# ==========================================

def main():
    rclpy.init()
    node = HW2Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()