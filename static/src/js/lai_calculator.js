/** @odoo-module */
$(document).on('submit', '.lai-upload-form', function () {
    const $btn = $(this).find('.lai-submit-btn');
    $btn.html('<span>Calculating…</span>').prop('disabled', true);
});