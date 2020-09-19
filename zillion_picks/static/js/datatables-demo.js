// Call the dataTables jQuery plugin
$(document).ready(function () {
	$("#dataTable").DataTable({
		paging: false,
		dom: "Bfrtip",
		// scrollY: '600',
		// scrollX: true,
		buttons: ["copy", "csv", "excel"],
		// responsive: true,
	});
});
