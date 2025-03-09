import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Ellipsis, Flag, Pencil, Trash2 } from "lucide-react";
import { useState } from "react";
import { CommentDeleteModal } from "./comment-delete-modal";
import { CommentEditModal } from "./comment-edit-modal";
import { CommentReportModal } from "./comment-report-modal";

interface CommentActionsMenuProps {
  movieId: number;
  commentId: string;
  content: string;
  isSpoiler: boolean;
  isOwner: boolean;
}

const CommentActionsMenu: React.FC<CommentActionsMenuProps> = ({
  movieId,
  commentId,
  content,
  isSpoiler,
  isOwner,
}) => {
  const [isMenuOpen, setMenuOpen] = useState(false);
  const [isReportModalOpen, setReportModalOpen] = useState(false);
  const [isEditModalOpen, setEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setDeleteModalOpen] = useState(false);

  const handleReportSuccess = () => {
    setReportModalOpen(false);
    setMenuOpen(false);
  };

  const handleEditSuccess = () => {
    setEditModalOpen(false);
    setMenuOpen(false);
  };

  const handleDeleteSuccess = () => {
    setDeleteModalOpen(false);
    setMenuOpen(false);
  };

  return (
    <DropdownMenu open={isMenuOpen} onOpenChange={setMenuOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon">
          <Ellipsis className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem
          onSelect={(e) => e.preventDefault()}
          onClick={() => setReportModalOpen(true)}
        >
          <Flag className="h-4 w-4" />
          Report
        </DropdownMenuItem>
        <CommentReportModal
          movieId={movieId}
          commentId={commentId}
          open={isReportModalOpen}
          onOpenChange={setReportModalOpen}
          onSuccess={handleReportSuccess}
        />
        {isOwner && (
          <>
            <DropdownMenuItem
              onSelect={(e) => e.preventDefault()}
              onClick={() => setEditModalOpen(true)}
            >
              <Pencil className="h-4 w-4" />
              Edit
            </DropdownMenuItem>
            <CommentEditModal
              movieId={movieId}
              commentId={commentId}
              content={content}
              isSpoiler={isSpoiler}
              open={isEditModalOpen}
              onOpenChange={setEditModalOpen}
              onSuccess={handleEditSuccess}
            />
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive"
              onSelect={(e) => e.preventDefault()}
              onClick={() => setDeleteModalOpen(true)}
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </DropdownMenuItem>
            <CommentDeleteModal
              movieId={movieId}
              commentId={commentId}
              open={isDeleteModalOpen}
              onOpenChange={setDeleteModalOpen}
              onSuccess={handleDeleteSuccess}
            />
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default CommentActionsMenu;
