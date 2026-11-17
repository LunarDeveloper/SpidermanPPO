extends SceneTree
## Verify cursor-directed casts independently from the tumbling forearm basis.
const Arena = preload("res://spider_environment.gd")

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var arena := Arena.new()
	root.add_child(arena)
	paused = true
	var target := Vector3(-26.5, 80, 35)
	var directions: Array = []
	for arm in arena.elbows:
		directions.append((target - arm.position).normalized())
		# The current physical orientation intentionally disagrees with the cursor.
		arm.rotation = Vector3(0.4, 1.2, -0.7)
	arena.apply_action([0, 0, 0, 0, 1, 1], 0.01, directions)
	for i in range(2):
		if not arena.attached[i] or arena.anchors[i].distance_to(target) > 0.001:
			push_error("Manual cast did not land at the preview target")
			quit(1)
			return
	if not arena.web_target(Vector3.ZERO, Vector3.UP).is_empty():
		push_error("Vertical aim should have no wall target")
		quit(1)
		return
	if not arena.web_target(Vector3(0, 55, 0), Vector3(0.01, 0, 1).normalized()).is_empty():
		push_error("Out-of-range aim should have no wall target")
		quit(1)
		return
	print("PASS: manual casts match wall target despite arm rotation; invalid aim rejected")
	quit()
