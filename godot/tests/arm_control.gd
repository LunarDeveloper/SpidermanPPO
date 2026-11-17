extends SceneTree
## Real-physics checks for aiming, observations, and terminal ground penalties.
const Arena = preload("res://spider_environment.gd")

func _initialize() -> void:
	call_deferred("_run")

func _check(condition: bool, message: String) -> bool:
	if not condition:
		push_error(message)
		quit(1)
	return condition

func _run() -> void:
	var arena := Arena.new()
	root.add_child(arena)
	await physics_frame
	# Both a forward/upward aim and a large backwards turn must converge.
	for yaw in [0.25, 0.8]:
		arena.reset_episode(0)
		var action := [-yaw, 0.4, yaw, 0.4, -1, -1]
		var targets := [arena._direction(-yaw, 0.4), arena._direction(yaw, 0.4)]
		var initial_error := arena._arm_direction(0).angle_to(targets[0])
		for step in range(150):
			arena.apply_action(action, 0.01)
			await physics_frame
			if not _check(arena.pelvis.position.is_finite(), "Nonfinite ragdoll position"):
				return
		for i in range(2):
			var error := arena._arm_direction(i).angle_to(targets[i])
			print("Aim yaw=", yaw, " arm=", i, " error_degrees=", rad_to_deg(error))
			if not _check(error < 0.25 and error < initial_error * 0.4, "Arm did not track its requested direction"):
				return
		var obs := arena.observation()
		if not _check(obs.size() == 20 and is_equal_approx(obs[1], arena.pelvis.position.y), "Height missing from observation"):
			return
		if not _check(Vector3(obs[2], obs[3], obs[4]).is_equal_approx(arena.pelvis.linear_velocity), "Velocity missing from observation"):
			return
		for i in range(2):
			var start := 5 + i * 3
			var observed := Vector3(obs[start], obs[start + 1], obs[start + 2])
			if not _check(observed.is_equal_approx(arena._arm_direction(i)), "Observation must report actual arm direction"):
				return
	# The antiparallel case must produce a bounded turn, rather than zero torque.
	var torque := arena._aim_torque(arena.elbows[0], -arena._arm_direction(0), -1.0)
	if not _check(torque.is_finite() and torque.length() > 0 and torque.length() <= Arena.ARM_MAX_TORQUE + 0.001, "Invalid 180-degree aiming torque"):
		return
	# Drop the full ragdoll just above the actual floor. Any limb may trigger it.
	arena.reset_episode(0)
	for body in arena.bodies:
		body.position.y -= 52.5
		body.linear_velocity = Vector3.ZERO
		PhysicsServer3D.body_set_state(body.get_rid(), PhysicsServer3D.BODY_STATE_TRANSFORM, body.transform)
		PhysicsServer3D.body_set_state(body.get_rid(), PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3.ZERO)
	var terminal: Dictionary = {}
	for step in range(100):
		await physics_frame
		terminal = arena.transition(0.01)
		if terminal.info.ground_contact:
			break
	if not _check(terminal.info.ground_contact and terminal.terminated, "Actual floor contact must terminate"):
		return
	if not _check(is_equal_approx(terminal.info.ground_penalty, 0.5) and terminal.reward < -0.49, "Missing small ground penalty"):
		return
	if not _check(is_zero_approx(arena.transition(0.01).info.ground_penalty), "Ground penalty repeated"):
		return
	arena.reset_episode(0)
	if not _check(not arena.ground_contact and not arena.ground_penalized, "Contact flags did not reset"):
		return
	arena._on_body_contact(arena.pelvis) # Non-ground bodies must not count.
	if not _check(not arena.ground_contact, "Non-ground contact was penalized"):
		return
	print("PASS: physical arm tracking, 180-degree turn, height/speed/direction observations, ground contact and reset")
	quit()
