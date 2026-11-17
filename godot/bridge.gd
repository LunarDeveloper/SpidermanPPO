extends Node3D
## Newline-delimited JSON on loopback only. One outstanding request per client.
const Arena = preload("res://spider_environment.gd")
var server := TCPServer.new()
var peer: StreamPeerTCP
var buffer := ""
var busy := false
var arena: Node3D
var port := 11000
var training := false
var instance_token := ""
var status: Label
var aim_overlay: Control
var aim_point := Vector3.ZERO
var aim_valid := false
var manual_paused := false
var slow_motion := false
var manual_directions: Array = [Vector3.ZERO, Vector3.ZERO]

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	var args := OS.get_cmdline_user_args()
	for i in range(args.size()):
		if args[i] == "--token" and i + 1 < args.size():
			instance_token = args[i + 1]
		if args[i] == "--port" and i + 1 < args.size():
			port = int(args[i + 1])
			training = true
	arena = Arena.new()
	arena.process_mode = Node.PROCESS_MODE_PAUSABLE
	add_child(arena)
	var canvas := CanvasLayer.new()
	add_child(canvas)
	status = Label.new()
	status.position = Vector2(20, 20)
	canvas.add_child(status)
	if training:
		var error := server.listen(port, "127.0.0.1")
		if error != OK:
			push_error("Cannot listen on training port %s: %s" % [port, error])
			get_tree().quit(1)
			return
		get_tree().paused = true
		print("SPIDER_READY ", port)
	else:
		aim_overlay = Control.new()
		aim_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		aim_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
		canvas.add_child(aim_overlay)
		aim_overlay.draw.connect(_draw_aim)
		print("Manual sandbox: mouse aim, LMB/RMB or Q/E webs, Space pause, T slow motion, R reset.")

func _process(_delta: float) -> void:
	status.text = "Godot Spider PPO | %s | distance %.1f m" % ["GPU trainer connection" if training else "Manual sandbox", arena.pelvis.position.z]
	if not training:
		_update_aim()
		var target_text := "Point at a wall within 140 m"
		if aim_valid:
			target_text = "Target %.0f m | Left: %s   Right: %s" % [arena.pelvis.position.distance_to(aim_point), "ATTACHED" if arena.attached[0] else "ready", "ATTACHED" if arena.attached[1] else "ready"]
		status.text += "\nMouse: aim   Hold LMB / Q: left web   Hold RMB / E: right web\nSpace: pause   T: slow motion   R: reset\n%s%s%s" % [target_text, " | PAUSED" if manual_paused else "", " | 0.25x speed" if slow_motion else ""]
		aim_overlay.queue_redraw()
		return
	if peer == null:
		if server.is_connection_available():
			peer = server.take_connection()
			peer.set_no_delay(true)
		return
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		peer = null
		buffer = ""
		get_tree().paused = true
		return
	var available := peer.get_available_bytes()
	if available > 0:
		var data := peer.get_data(available)
		if data[0] != OK:
			return
		buffer += data[1].get_string_from_utf8()
	if buffer.length() > 65536:
		peer.disconnect_from_host()
		return
	if busy or not buffer.contains("\n"):
		return
	var split := buffer.find("\n")
	var request = JSON.parse_string(buffer.substr(0, split))
	buffer = buffer.substr(split + 1)
	if not request is Dictionary:
		_reply({"error": "Expected a JSON object"})
		return
	_handle(request)

func _handle(request: Dictionary) -> void:
	busy = true
	match request.get("command", ""):
		"hello":
			_reply({"protocol": 1, "token": instance_token, "observations": 20, "actions": 6})
		"reset":
			arena.reset_episode(int(request.get("seed", 0)))
			# Flush reset transforms through the physics server before observing.
			await get_tree().physics_frame
			_reply({"observation": arena.observation(), "info": {"physics_dt": 0.01}})
		"step":
			var action = request.get("action", [])
			if not action is Array or action.size() != 6:
				_reply({"error": "Expected six finite actions in [-1, 1]"})
				busy = false
				return
			for value in action:
				if not (value is float or value is int) or not is_finite(float(value)):
					_reply({"error": "Actions must be finite numbers"})
					busy = false
					return
			# The first signal occurs before physics. Apply forces there, then
			# observe at the next signal, after exactly one physics interval.
			await get_tree().physics_frame
			get_tree().paused = false
			arena.apply_action(action, 0.01)
			await get_tree().physics_frame
			get_tree().paused = true
			_reply(arena.transition(0.01))
		"close":
			_reply({"ok": true})
			get_tree().quit()
		_:
			_reply({"error": "Unknown command"})
	busy = false

func _reply(response: Dictionary) -> void:
	if peer != null and peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		peer.put_data((JSON.stringify(response) + "\n").to_utf8_buffer())

func _physics_process(delta: float) -> void:
	if training or manual_paused:
		return
	_update_aim()
	arena.apply_action([0, 0, 0, 0,
		1.0 if Input.is_physical_key_pressed(KEY_Q) or Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT) else -1.0,
		1.0 if Input.is_physical_key_pressed(KEY_E) or Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT) else -1.0], delta, manual_directions)
	arena.transition(delta)

func _unhandled_input(event: InputEvent) -> void:
	if training or not event is InputEventKey or not event.pressed or event.echo:
		return
	match event.physical_keycode:
		KEY_R:
			arena.reset_episode(0)
		KEY_SPACE:
			manual_paused = not manual_paused
			get_tree().paused = manual_paused
		KEY_T:
			slow_motion = not slow_motion
			Engine.time_scale = 0.25 if slow_motion else 1.0

func _update_aim() -> void:
	var mouse := get_viewport().get_mouse_position()
	var origin: Vector3 = arena.camera.project_ray_origin(mouse)
	var direction: Vector3 = arena.camera.project_ray_normal(mouse)
	var hit: Dictionary = arena.web_target(origin, direction)
	aim_valid = not hit.is_empty()
	manual_directions = [Vector3.ZERO, Vector3.ZERO]
	if not aim_valid:
		return
	aim_point = hit.point
	for i in range(2):
		var arm_position: Vector3 = arena.elbows[i].position
		var shot := (aim_point - arm_position).normalized()
		if arena.web_target(arm_position, shot).is_empty():
			aim_valid = false
		else:
			manual_directions[i] = shot

func _draw_aim() -> void:
	var mouse := get_viewport().get_mouse_position()
	var color := Color(0.3, 1.0, 0.5) if aim_valid else Color(1.0, 0.4, 0.3)
	aim_overlay.draw_arc(mouse, 9, 0, TAU, 32, color, 2, true)
	for axis in [Vector2.RIGHT, Vector2.DOWN]:
		aim_overlay.draw_line(mouse - axis * 15, mouse - axis * 5, color, 2, true)
		aim_overlay.draw_line(mouse + axis * 5, mouse + axis * 15, color, 2, true)
	if not aim_valid:
		return
	var target: Vector2 = arena.camera.unproject_position(aim_point)
	aim_overlay.draw_circle(target, 4, color)
	for i in range(2):
		var arm_position: Vector3 = arena.elbows[i].position
		if not arena.camera.is_position_behind(arm_position):
			var start: Vector2 = arena.camera.unproject_position(arm_position)
			aim_overlay.draw_dashed_line(start, target, Color(color, 0.45), 1, 6, true)
