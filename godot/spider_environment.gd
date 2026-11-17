extends Node3D
## +Z is progress, +Y is up, matching the original policy interface.
const GAP := 53.0
const HEIGHT := 120.0
const START := Vector3(0, 55, 0)
const MAX_STEPS := 3000
const ARM_AIM_STIFFNESS := 120.0
const ARM_AIM_DAMPING := 22.0
const ARM_MAX_ACCELERATION := 200.0
const ARM_MAX_TORQUE := 6.0
const GROUND_CONTACT_PENALTY := 0.5
var pelvis: RigidBody3D
var chest: RigidBody3D
var upper_arms: Array[RigidBody3D] = []
var elbows: Array[RigidBody3D] = []
var aim_targets: Array[Vector3] = [Vector3.LEFT, Vector3.RIGHT]
var bodies: Array[RigidBody3D] = []
var initial: Array[Transform3D] = []
var anchors := [Vector3.ZERO, Vector3.ZERO]
var attached := [false, false]
var lengths := [0.0, 0.0]
var cooldowns := [0.0, 0.0]
var new_attachments := 0
var current_distance := 0.0
var steps := 0
var episode_return := 0.0
var camera: Camera3D
var web_meshes: Array[MeshInstance3D] = []
var street: Node3D
var ground_contact := false
var ground_penalized := false
var visuals_enabled := true

func _ready() -> void:
	_build_world()
	pelvis = _body("Pelvis", START, Vector3(0.65, 0.7, 0.4), 8.0)
	chest = _body("Chest", START + Vector3(0, 0.65, 0), Vector3(0.9, 0.7, 0.4), 8.0)
	_joint(pelvis, chest, START + Vector3(0, 0.35, 0))
	var head := _body("Head", START + Vector3(0, 1.3, 0), Vector3(0.4, 0.4, 0.4), 2.0)
	_joint(chest, head, START + Vector3(0, 1.05, 0))
	for side in [-1.0, 1.0]:
		var upper := _body("UpperArm", START + Vector3(side * 0.7, 0.65, 0), Vector3(0.55, 0.22, 0.22), 1.5)
		var forearm := _body("Forearm", START + Vector3(side * 1.25, 0.65, 0), Vector3(0.55, 0.2, 0.2), 1.0)
		_joint(chest, upper, START + Vector3(side * 0.45, 0.65, 0))
		_joint(upper, forearm, START + Vector3(side * 0.98, 0.65, 0))
		upper_arms.append(upper)
		elbows.append(forearm)
		var thigh := _body("Thigh", START + Vector3(side * 0.23, -0.65, 0), Vector3(0.28, 0.7, 0.28), 4.0)
		var shin := _body("Shin", START + Vector3(side * 0.23, -1.35, 0), Vector3(0.24, 0.7, 0.24), 3.0)
		_joint(pelvis, thigh, START + Vector3(side * 0.23, -0.3, 0))
		_joint(thigh, shin, START + Vector3(side * 0.23, -1.0, 0))
		if not visuals_enabled:
			continue
		var web := MeshInstance3D.new()
		var material := StandardMaterial3D.new()
		material.albedo_color = Color.WHITE
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		web.material_override = material
		add_child(web)
		web_meshes.append(web)
	reset_episode(0)
	set_process(visuals_enabled)

func _body(label: String, at: Vector3, size: Vector3, mass_value: float) -> RigidBody3D:
	var body := RigidBody3D.new()
	body.name = label
	body.mass = mass_value
	body.collision_layer = 2
	body.collision_mask = 1 # No ragdoll self-collisions.
	body.can_sleep = false
	body.continuous_cd = true
	body.linear_damp = 0.02
	body.angular_damp = 0.2
	body.contact_monitor = true
	body.max_contacts_reported = 8
	body.body_entered.connect(_on_body_contact)
	# Explicit box inertia lets the servo request comparable angular acceleration
	# for upper arms and forearms, independent of their different masses and sizes.
	body.inertia = mass_value / 12.0 * Vector3(
		size.y * size.y + size.z * size.z,
		size.x * size.x + size.z * size.z,
		size.x * size.x + size.y * size.y)
	add_child(body)
	body.position = at
	var collision := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = size
	collision.shape = shape
	body.add_child(collision)
	if visuals_enabled:
		var visual := MeshInstance3D.new()
		var mesh := BoxMesh.new()
		mesh.size = size
		visual.mesh = mesh
		var material := StandardMaterial3D.new()
		material.albedo_color = Color(0.8, 0.08, 0.12) if mass_value >= 8 else Color(0.06, 0.15, 0.6)
		visual.material_override = material
		body.add_child(visual)
	bodies.append(body)
	initial.append(body.transform)
	return body

func _joint(a: RigidBody3D, b: RigidBody3D, at: Vector3) -> void:
	var joint := PinJoint3D.new()
	add_child(joint)
	joint.position = at
	joint.node_a = joint.get_path_to(a)
	joint.node_b = joint.get_path_to(b)
	joint.set_param(PinJoint3D.PARAM_BIAS, 0.3)

func _block(parent: Node3D, at: Vector3, size: Vector3, color: Color, solid: bool, ground: bool = false) -> void:
	if visuals_enabled:
		var block := MeshInstance3D.new()
		var mesh := BoxMesh.new()
		mesh.size = size
		block.mesh = mesh
		var material := StandardMaterial3D.new()
		material.albedo_color = color
		block.material_override = material
		parent.add_child(block)
		block.position = at
	if solid:
		var body := StaticBody3D.new()
		parent.add_child(body)
		if ground:
			body.add_to_group("ground")
		body.position = at
		var collision := CollisionShape3D.new()
		var shape := BoxShape3D.new()
		shape.size = size
		collision.shape = shape
		body.add_child(collision)

func _build_world() -> void:
	if visuals_enabled:
		var sun := DirectionalLight3D.new()
		add_child(sun)
		sun.rotation_degrees = Vector3(-40, -30, 0)
		var environment := WorldEnvironment.new()
		environment.environment = Environment.new()
		environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
		environment.environment.ambient_light_color = Color(0.7, 0.8, 1)
		environment.environment.ambient_light_energy = 0.6
		add_child(environment)
	street = Node3D.new()
	add_child(street)
	# Continuous physical walls make the analytical web attachment geometry exact.
	_block(street, Vector3(0, -1, 0), Vector3(GAP, 2, 800), Color(0.12, 0.13, 0.15), true, true)
	for side in [-1, 1]:
		_block(street, Vector3(side * (GAP / 2 + 10), HEIGHT / 2, 0), Vector3(20, HEIGHT, 800), Color(0.27, 0.32, 0.4), true)
		for z in range(-360, 400, 30):
			_block(street, Vector3(side * (GAP / 2 - 0.02), HEIGHT / 2, z), Vector3(0.05, HEIGHT, 0.6), Color(0.1, 0.15, 0.22), false)
	if visuals_enabled:
		camera = Camera3D.new()
		camera.far = 1000
		add_child(camera)
		camera.current = true

func reset_episode(_seed: int) -> void:
	for i in range(bodies.size()):
		bodies[i].freeze = false
		bodies[i].transform = initial[i]
		bodies[i].linear_velocity = Vector3(0, 0, 12)
		bodies[i].angular_velocity = Vector3.ZERO
		PhysicsServer3D.body_set_state(bodies[i].get_rid(), PhysicsServer3D.BODY_STATE_TRANSFORM, initial[i])
		PhysicsServer3D.body_set_state(bodies[i].get_rid(), PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3(0, 0, 12))
		PhysicsServer3D.body_set_state(bodies[i].get_rid(), PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
		bodies[i].reset_physics_interpolation()
	attached = [false, false]
	anchors = [Vector3.ZERO, Vector3.ZERO]
	lengths = [0.0, 0.0]
	cooldowns = [0.0, 0.0]
	new_attachments = 0
	aim_targets = [Vector3.LEFT, Vector3.RIGHT]
	ground_contact = false
	ground_penalized = false
	current_distance = 0.0
	steps = 0
	episode_return = 0.0
	street.position.z = 0

func _direction(yaw: float, pitch: float) -> Vector3:
	return Vector3(sin(yaw * PI) * cos(pitch * PI / 2), sin(pitch * PI / 2), cos(yaw * PI) * cos(pitch * PI / 2))

func _arm_direction(index: int) -> Vector3:
	return elbows[index].global_basis.x * (-1.0 if index == 0 else 1.0)

func _aim_torque(body: RigidBody3D, target: Vector3, side: float) -> Vector3:
	if target.length_squared() < 0.000001:
		return Vector3.ZERO
	var current := (body.global_basis.x * side).normalized()
	var desired := target.normalized()
	var axis := current.cross(desired)
	var sine := axis.length()
	var cosine := clampf(current.dot(desired), -1, 1)
	var angle := atan2(sine, cosine)
	if sine > 0.00001:
		axis /= sine
	elif cosine < 0:
		# The cross product vanishes at 180 degrees; still choose a turn axis.
		axis = current.cross(Vector3.UP)
		if axis.length_squared() < 0.00001:
			axis = current.cross(Vector3.FORWARD)
		axis = axis.normalized()
	else:
		axis = Vector3.ZERO
	# Dampen only angular motion that changes pointing direction, leaving twist free.
	var angular_velocity := body.angular_velocity - current * body.angular_velocity.dot(current)
	var acceleration := (axis * angle * ARM_AIM_STIFFNESS - angular_velocity * ARM_AIM_DAMPING).limit_length(ARM_MAX_ACCELERATION)
	var basis := body.global_basis.orthonormalized()
	var local_acceleration := basis.transposed() * acceleration
	return (basis * (body.inertia * local_acceleration)).limit_length(ARM_MAX_TORQUE)

func _steer_arm(index: int, target: Vector3) -> void:
	var side := -1.0 if index == 0 else 1.0
	var upper := upper_arms[index]
	var forearm := elbows[index]
	var shoulder_torque := _aim_torque(upper, target, side)
	var elbow_torque := _aim_torque(forearm, target, side)
	upper.apply_torque(shoulder_torque - elbow_torque)
	forearm.apply_torque(elbow_torque)
	chest.apply_torque(-shoulder_torque)

func _on_body_contact(other: Node) -> void:
	if other.is_in_group("ground"):
		ground_contact = true

func apply_action(action: Array, delta: float, manual_directions: Array = []) -> void:
	# Physics scenery must follow progress even without visual _process callbacks.
	street.position.z = floorf(pelvis.position.z / 100) * 100
	new_attachments = 0
	for i in range(2):
		var target := _direction(clampf(action[i * 2], -1, 1), clampf(action[i * 2 + 1], -1, 1))
		if manual_directions.size() == 2:
			target = manual_directions[i]
		aim_targets[i] = target
		var arm := elbows[i]
		_steer_arm(i, target)
		cooldowns[i] = maxf(0, cooldowns[i] - delta)
		if action[4 + i] <= 0:
			if attached[i]:
				attached[i] = false
				cooldowns[i] = 0.25
		elif not attached[i] and cooldowns[i] <= 0:
			_attach(i, target if manual_directions.size() == 2 else _arm_direction(i))
		if attached[i]:
			lengths[i] = maxf(0, lengths[i] - 3 * delta)
			var offset: Vector3 = anchors[i] - arm.position
			var distance := offset.length()
			if distance > 0.001:
				var direction := offset / distance
				# Tension-only spring: a web cannot push the arm away.
				var tension := maxf(0, 200 * (distance - lengths[i]) - 5 * arm.linear_velocity.dot(direction))
				arm.apply_central_force(direction * minf(tension, 20000))

func web_target(origin: Vector3, direction: Vector3) -> Dictionary:
	# Shared by cursor previews and actual casts so range/height checks agree.
	if absf(direction.x) < 0.001:
		return {}
	var wall := GAP / 2 * signf(direction.x)
	var distance := (wall - origin.x) / direction.x
	if distance <= 0 or distance > 140:
		return {}
	var point := origin + direction * distance
	if point.y < 0 or point.y > HEIGHT:
		return {}
	return {"point": point, "distance": distance}

func _attach(i: int, direction: Vector3) -> void:
	var hit := web_target(elbows[i].position, direction)
	if hit.is_empty():
		return
	anchors[i] = hit.point
	lengths[i] = hit.distance
	attached[i] = true
	new_attachments += 1

func observation() -> Array:
	var p := pelvis.position
	var v := pelvis.linear_velocity
	var left := _arm_direction(0)
	var right := _arm_direction(1)
	var la: Vector3 = anchors[0] - p if attached[0] else Vector3.ZERO
	var ra: Vector3 = anchors[1] - p if attached[1] else Vector3.ZERO
	# Attachment vectors now describe the world anchor, rather than the original
	# implementation's local elbow anchor, so the policy can observe web geometry.
	# p.y is current height; v contains signed XYZ speed. Unit arm directions
	# encode yaw/pitch continuously, without angle wrapping at +/- pi.
	return [p.x, p.y, v.x, v.y, v.z, left.x, left.y, left.z, right.x, right.y, right.z,
		1.0 if attached[0] else -1.0, 1.0 if attached[1] else -1.0,
		p.z - current_distance, la.x, la.y, la.z, ra.x, ra.y, ra.z]

func transition(delta: float) -> Dictionary:
	steps += 1
	current_distance = maxf(current_distance, pelvis.position.z) + 0.5 * delta
	var height_bonus := lerpf(0.5, 1, clampf(pelvis.position.y / HEIGHT, 0, 1))
	var center_bonus := clampf((GAP - absf(pelvis.position.x)) / GAP, 0.8, 1)
	# Clamp before fractional pow to avoid NaNs for backwards velocity.
	var velocity_bonus := pow(maxf(pelvis.linear_velocity.z, 0), 1.02) * 0.25
	# Web attachments are free; reward forward movement and penalize ground contact.
	var reward := velocity_bonus * height_bonus * center_bonus * delta
	var ground_penalty := 0.0
	if ground_contact and not ground_penalized:
		ground_penalty = GROUND_CONTACT_PENALTY
		ground_penalized = true
		reward -= ground_penalty
	episode_return += reward
	# Contact with any limb ends the episode; don't end at height 2 before the
	# feet reach the street. The below-ground fallback handles missed contacts.
	var terminated := ground_contact or pelvis.position.y < 0 or pelvis.position.z < current_distance - 2
	var truncated := steps >= MAX_STEPS
	var info := {"distance": pelvis.position.z, "steps": steps, "attachments": new_attachments,
		"height": pelvis.position.y, "speed": pelvis.linear_velocity.length(),
		"forward_speed": pelvis.linear_velocity.z, "ground_contact": ground_contact,
		"ground_penalty": ground_penalty,
		"left_aim_error_degrees": rad_to_deg(_arm_direction(0).angle_to(aim_targets[0])) if not aim_targets[0].is_zero_approx() else 0.0,
		"right_aim_error_degrees": rad_to_deg(_arm_direction(1).angle_to(aim_targets[1])) if not aim_targets[1].is_zero_approx() else 0.0}
	return {"observation": observation(), "reward": reward, "terminated": terminated,
		"truncated": truncated, "info": info}

func _process(_delta: float) -> void:
	if pelvis == null:
		return
	# Move the continuous corridor in whole chunks; anchor coordinates stay global.
	street.position.z = floorf(pelvis.position.z / 100) * 100
	camera.position = pelvis.position + Vector3(8, 5, -14)
	camera.look_at(pelvis.position + Vector3(0, 1, 8))
	for i in range(2):
		web_meshes[i].visible = attached[i]
		if attached[i]:
			var mesh := ImmediateMesh.new()
			mesh.surface_begin(Mesh.PRIMITIVE_LINES)
			mesh.surface_add_vertex(elbows[i].position)
			mesh.surface_add_vertex(anchors[i])
			mesh.surface_end()
			web_meshes[i].mesh = mesh
