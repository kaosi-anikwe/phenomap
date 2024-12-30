// Show toasts
var toastElList = [].slice.call(document.querySelectorAll(".toast"));
var toastList = toastElList.map(function (toastEl) {
  return new bootstrap.Toast(toastEl, { delay: 5000 });
});
toastList.forEach((toast) => toast.show());

const flashMessage = (message, catetgory = "info", delay = 5000) => {
  const flashContainer = document.getElementById("flash-container");
  let newToast = document.createElement("div");
  newToast.setAttribute(
    "class",
    `toast align-items-center text-white bg-${catetgory} border-0`
  );
  newToast.innerHTML = `
            <div class="d-flex">
              <div class="toast-body">${message}</div>
                <button
                  type="button"
                  class="mx-2 close"
                  data-dismiss="toast"
                  aria-label="Close"
                >
                  <span aria-hidden="true">&times;</span>
                </button>
            </div>
            `;
  flashContainer.appendChild(newToast);
  newToast = new bootstrap.Toast(newToast, { delay: delay });
  newToast.show();
};
