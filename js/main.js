function toggle(id) {
  var x = document.getElementById(id);
  x.classList.toggle("show");
}

var clipboard = new ClipboardJS('.copy-code-button');

clipboard.on('success', function(e) {
	e.trigger.classList.add("copied");
	setTimeout(() => {
		e.trigger.classList.remove("copied");
	}, 1000);
});