document.addEventListener("DOMContentLoaded", function () {
  const casesTable = new DataTable("#cases-table", {
    paging: false,
    info: false,
    ajax: "/api/cases/",
    columns: [
      {
        data: null,
        orderable: false,
        render: function (data) {
          if (data.image) {
            return `
                <img src="/api/images/${data.image}" alt="Case #${data.uid}">
            `;
          }
          return `
              <div class="image-upload-icon d-flex">
                <i
                  class="typcn typcn-user text-primary border-primary rounded-circle"
                  style="font-size: 48px"
                ></i>
              </div>
          `;
        },
      },
      {
        data: "uid",
        render: function (data) {
          return `#${data}`;
        },
      },
      { data: "name" },
      {
        data: "image_count",
        className: "text-center",
      },
      {
        data: "created",
      },
      {
        data: "modified",
      },
      {
        data: "status",
      },
      {
        data: null,
        orderable: false,
        render: function (data) {
          return `
            <div class="btn-group">
                <button class="btn btn-sm btn-outline-secondary" onclick="editCase('${data.uid}')">
                    <i class="ti-pencil"></i> Edit
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteCase('${data.uid}')">
                    <i class="ti-trash"></i> Delete
                </button>
            </div>
        `;
        },
      },
    ],
  });
});

// Case actions
function editCase(caseId) {
  window.location.href = `/case/${caseId}`;
}

// Delete Case
const deleteCase = async (uid) => {
  try {
    if (confirm("This case and all its records will be deleted. Proceed?")) {
      window.location.href = `/case/delete/${uid}`;
    }
  } catch (error) {
    console.error(error);
    flashMessage(
      "Something went wrong. Please refresh the page and try again.",
      "danger"
    );
  }
};
