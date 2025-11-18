/** @odoo-module alias=lai_estimator.lai_calculator **/
$(document).ready(function () {
    $('.lai-upload-form').on('submit', function () {
        const $btn = $(this).find('.lai-submit-btn');
        $btn.html('<span>Calculating…</span>').prop('disabled', true);
    });
});