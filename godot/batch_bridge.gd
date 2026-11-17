extends Node
## Protocol 2: one action/observation batch per process. Worlds never collide.
const Arena = preload("res://spider_environment.gd")
const DT := 0.01
var arenas: Array[Node3D] = []
var server := TCPServer.new()
var peer: StreamPeerTCP
var buffer := ""
var busy := false
var port := 11000
var token := ""
var count := 8
var repeat := 4
var tick_gamma := 0.99

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	var args := OS.get_cmdline_user_args()
	for i in range(args.size() - 1):
		match args[i]:
			"--port": port = int(args[i + 1])
			"--token": token = args[i + 1]
			"--arenas": count = int(args[i + 1])
			"--repeat": repeat = int(args[i + 1])
			"--tick-gamma": tick_gamma = float(args[i + 1])
	if count < 1 or count > 256 or repeat not in [1, 4, 5] or tick_gamma <= 0 or tick_gamma > 1:
		push_error("Invalid batch configuration")
		get_tree().quit(1)
		return
	var visible := DisplayServer.get_name() != "headless"
	for i in range(count):
		var viewport := SubViewport.new()
		viewport.own_world_3d = true
		viewport.size = Vector2i(1280, 720) if visible and i == 0 else Vector2i(2, 2)
		viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS if visible and i == 0 else SubViewport.UPDATE_DISABLED
		if visible and i == 0:
			var container := SubViewportContainer.new()
			container.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
			container.stretch = true
			add_child(container)
			container.add_child(viewport)
		else:
			add_child(viewport)
		var arena := Arena.new()
		arena.visuals_enabled = visible and i == 0
		arena.process_mode = Node.PROCESS_MODE_PAUSABLE
		viewport.add_child(arena)
		arenas.append(arena)
	get_tree().paused = true
	var error := server.listen(port, "127.0.0.1")
	if error != OK:
		push_error("Cannot listen on batch port %s: %s" % [port, error])
		get_tree().quit(1)
	else:
		print("SPIDER_BATCH_READY ", port, " arenas=", count, " repeat=", repeat)

func _process(_delta: float) -> void:
	if peer == null:
		if server.is_connection_available():
			peer = server.take_connection()
			peer.set_no_delay(true)
		return
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		# A worker belongs to exactly one trainer; don't leak idle orphan processes.
		get_tree().paused = true
		get_tree().quit()
		return
	var available := peer.get_available_bytes()
	if available:
		var data := peer.get_data(available)
		if data[0] == OK:
			buffer += data[1].get_string_from_utf8()
	if buffer.length() > 1048576:
		peer.disconnect_from_host()
		return
	if busy or not buffer.contains("\n"):
		return
	var end := buffer.find("\n")
	var request = JSON.parse_string(buffer.substr(0, end))
	buffer = buffer.substr(end + 1)
	if not request is Dictionary:
		_reply({"error": "Expected object"})
		return
	_handle(request)

func _handle(request: Dictionary) -> void:
	busy = true
	match request.get("command", ""):
		"hello":
			var spaces: Array = []
			for arena in arenas:
				spaces.append(arena.get_world_3d().space.get_id())
			_reply({"protocol": 2, "token": token, "arenas": count, "repeat": repeat,
				"tick_gamma": tick_gamma, "physics_dt": DT, "observations": 20, "actions": 6, "spaces": spaces})
		"reset":
			for arena in arenas:
				arena.reset_episode(0)
			await get_tree().physics_frame
			var observations: Array = []
			for arena in arenas:
				observations.append(arena.observation())
			_reply({"observations": observations})
		"step":
			var actions = request.get("actions", [])
			if not _valid_actions(actions):
				_reply({"error": "Expected arenas x 6 finite actions"})
			else:
				await _step_batch(actions)
		"close":
			_reply({"ok": true})
			get_tree().quit()
		_:
			_reply({"error": "Unknown command"})
	busy = false

func _valid_actions(actions: Variant) -> bool:
	if not actions is Array or actions.size() != count:
		return false
	for action in actions:
		if not action is Array or action.size() != 6:
			return false
		for value in action:
			if not (value is int or value is float) or not is_finite(float(value)):
				return false
	return true

func _step_batch(actions: Array) -> void:
	var started := Time.get_ticks_usec()
	var script_usec := 0
	var results: Array = []
	for i in range(count):
		results.append({"reward": 0.0, "raw_reward": 0.0, "terminated": false,
			"truncated": false, "physics_steps": 0, "attachments": 0, "ground_penalty": 0.0})
	await get_tree().physics_frame
	get_tree().paused = false
	for substep in range(repeat):
		var work_started := Time.get_ticks_usec()
		for i in range(count):
			if not results[i].terminated and not results[i].truncated:
				arenas[i].apply_action(actions[i], DT)
		script_usec += Time.get_ticks_usec() - work_started
		await get_tree().physics_frame
		work_started = Time.get_ticks_usec()
		for i in range(count):
			var result: Dictionary = results[i]
			if result.terminated or result.truncated:
				continue
			var tick: Dictionary = arenas[i].transition(DT)
			result.reward += pow(tick_gamma, substep) * tick.reward
			result.raw_reward += tick.reward
			result.physics_steps += 1
			result.terminated = tick.terminated
			result.truncated = tick.truncated
			result.info = tick.info
			result.attachments += tick.info.attachments
			result.ground_penalty += tick.info.ground_penalty
			if tick.terminated or tick.truncated:
				result.terminal_observation = tick.observation
				# No remaining physics or rewards from this episode within the batch.
				for body in arenas[i].bodies:
					body.freeze = true
		script_usec += Time.get_ticks_usec() - work_started
	get_tree().paused = true
	for i in range(count):
		if results[i].terminated or results[i].truncated:
			arenas[i].reset_episode(0)
		results[i].observation = arenas[i].observation()
		results[i].info.attachments = results[i].attachments
		results[i].info.ground_penalty = results[i].ground_penalty
		if arenas[i].visuals_enabled:
			arenas[i]._process(0.0)
	_reply({"results": results, "timing": {"batch_s": (Time.get_ticks_usec() - started) / 1000000.0,
		"script_s": script_usec / 1000000.0}})

func _reply(response: Dictionary) -> void:
	if peer != null and peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		peer.put_data((JSON.stringify(response) + "\n").to_utf8_buffer())
